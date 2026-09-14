"""Agentic Orchestrator — ReAct agent that coordinates all AARAG layers."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .tools import ToolRegistry, Tool, create_default_tools
from .planner import MultiStepPlanner, Plan, PlanStep


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace."""
    thought: str
    action: str
    action_input: str
    observation: str
    step_number: int


@dataclass
class AgentResponse:
    """The agent's final response."""
    answer: str
    reasoning_trace: List[AgentStep]
    confidence: float
    tools_used: List[str]
    total_steps: int
    strategy_used: str


class AgenticOrchestrator:
    """ReAct-based agent that orchestrates all AARAG layers.

    Based on: ReAct (Yao et al., ICLR 2023)

    The agent reasons about which tools to use, takes actions,
    observes results, and iterates until it can answer the query.
    """

    def __init__(
        self,
        llm_fn=None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        planning_strategy: str = "react",
    ):
        self.llm_fn = llm_fn
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.planning_strategy = planning_strategy
        self.planner = MultiStepPlanner(llm_fn=llm_fn, max_steps=max_steps)

    def answer(
        self,
        query: str,
        routing_strategy: str = "multi_step",
        context: Optional[str] = None,
    ) -> AgentResponse:
        """Main entry point: answer a query using the agentic loop.

        Args:
            query: The user's question
            routing_strategy: Strategy from the adaptive router
            context: Optional pre-retrieved context

        Returns:
            AgentResponse with answer and reasoning trace
        """
        if self.planning_strategy == "react":
            return self._react_loop(query, routing_strategy, context)
        elif self.planning_strategy == "plan_and_execute":
            return self._plan_and_execute(query, routing_strategy, context)
        else:
            return self._react_loop(query, routing_strategy, context)

    def _react_loop(
        self,
        query: str,
        routing_strategy: str,
        context: Optional[str],
    ) -> AgentResponse:
        """ReAct loop: Think -> Act -> Observe -> Repeat."""
        trace: List[AgentStep] = []
        tools_used: List[str] = []
        accumulated_context = context or ""
        step = 0

        while step < self.max_steps:
            step += 1

            # THINK: Decide what to do
            thought = self._think(query, routing_strategy, trace, accumulated_context)

            # Check if we should give final answer
            if "final_answer" in thought.lower() or "answer:" in thought.lower():
                break

            # ACT: Choose and execute a tool
            action, action_input = self._choose_action(thought, routing_strategy, query)

            if action == "final_answer":
                break

            # OBSERVE: Get tool result
            observation = self._execute_tool(action, action_input)
            if action not in tools_used:
                tools_used.append(action)

            # Record step
            trace.append(AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation[:500],
                step_number=step,
            ))

            # Update accumulated context
            if action in ["retrieve", "web_search", "graph_query"]:
                accumulated_context += f"\n\n{observation}"

        # Generate final answer
        final_answer = self._generate_final_answer(query, accumulated_context, trace)

        # Compute confidence
        confidence = self._compute_confidence(trace, tools_used)

        return AgentResponse(
            answer=final_answer,
            reasoning_trace=trace,
            confidence=confidence,
            tools_used=tools_used,
            total_steps=step,
            strategy_used=routing_strategy,
        )

    def _plan_and_execute(
        self,
        query: str,
        routing_strategy: str,
        context: Optional[str],
    ) -> AgentResponse:
        """Plan-then-execute: Create plan first, then execute steps."""
        plan = self.planner.create_plan(query, routing_strategy)

        trace: List[AgentStep] = []
        tools_used: List[str] = []
        accumulated_context = context or ""

        for plan_step in plan.steps:
            plan_step.status = "in_progress"

            # Execute the planned tool
            if plan_step.tool == "generate":
                plan_step.result = "Generation deferred to final answer."
                plan_step.status = "completed"
                continue

            observation = self._execute_tool(plan_step.tool, plan_step.input_query)
            plan_step.result = observation[:500]
            plan_step.status = "completed"

            if plan_step.tool not in tools_used:
                tools_used.append(plan_step.tool)

            trace.append(AgentStep(
                thought=f"Plan step {plan_step.step_id}: {plan_step.description}",
                action=plan_step.tool,
                action_input=plan_step.input_query,
                observation=observation[:500],
                step_number=plan_step.step_id,
            ))

            if plan_step.tool in ["retrieve", "web_search", "graph_query"]:
                accumulated_context += f"\n\n{observation}"

        # Generate final answer
        final_answer = self._generate_final_answer(query, accumulated_context, trace)
        confidence = self._compute_confidence(trace, tools_used)

        return AgentResponse(
            answer=final_answer,
            reasoning_trace=trace,
            confidence=confidence,
            tools_used=tools_used,
            total_steps=len(plan.steps),
            strategy_used=f"plan_and_execute ({routing_strategy})",
        )

    def _think(
        self,
        query: str,
        strategy: str,
        trace: List[AgentStep],
        context: str,
    ) -> str:
        """Generate reasoning about next action."""
        if self.llm_fn:
            trace_summary = "\n".join(
                f"Step {s.step_number}: Used {s.action} -> {s.observation[:100]}..."
                for s in trace[-3:]
            ) if trace else "No steps yet."

            prompt = f"""You are an intelligent agent answering questions using tools.

Question: {query}
Strategy: {strategy}
Previous steps:
{trace_summary}
Accumulated context length: {len(context)} chars

Available tools: retrieve, graph_query, web_search, decompose, verify

Think about what to do next. If you have enough information, say "FINAL_ANSWER: [your answer]".
Otherwise, describe what tool to use next.

Thought:"""
            return self.llm_fn(prompt).strip()

        # Heuristic thinking
        if not trace:
            if strategy == "graph_global":
                return "I should query the knowledge graph first for entity relationships."
            elif strategy == "multi_step":
                return "I should decompose this query and retrieve documents."
            else:
                return "I should retrieve relevant documents."

        if len(trace) >= 3:
            return "I have gathered enough information. FINAL_ANSWER."

        return f"I should continue with the {strategy} strategy."

    def _choose_action(self, thought: str, strategy: str, query: str) -> tuple:
        """Choose an action based on the thought."""
        thought_lower = thought.lower()

        # Check for tool mentions in thought
        for tool_name in ["graph_query", "web_search", "decompose", "verify", "retrieve"]:
            if tool_name in thought_lower:
                return tool_name, query

        if "final_answer" in thought_lower:
            return "final_answer", ""

        # Default action based on strategy
        if strategy == "graph_global":
            return "graph_query", query
        elif strategy == "multi_step":
            return "retrieve", query
        elif strategy == "single_step":
            return "retrieve", query
        else:
            return "final_answer", ""

    def _execute_tool(self, action: str, action_input: str) -> str:
        """Execute a tool and return the observation."""
        if self.tool_registry is None:
            return f"Tool '{action}' not available."

        try:
            result = self.tool_registry.execute(action, query=action_input)
            return str(result)
        except Exception as e:
            return f"Error executing {action}: {e}"

    def _generate_final_answer(
        self,
        query: str,
        context: str,
        trace: List[AgentStep],
    ) -> str:
        """Generate the final answer from accumulated context."""
        if self.llm_fn:
            trace_summary = ", ".join(s.action for s in trace)
            prompt = f"""Answer the following question based on the gathered context.

Question: {query}
Context: {context[:3000]}
Tools used: {trace_summary}

Provide a clear, accurate, and complete answer.

Answer:"""
            return self.llm_fn(prompt).strip()

        # Fallback: return context summary
        if context:
            return f"Based on the retrieved information: {context[:500]}"
        return f"I was unable to find sufficient information to answer: {query}"

    def _compute_confidence(self, trace: List[AgentStep], tools_used: List[str]) -> float:
        """Compute confidence based on the reasoning trace."""
        if not trace:
            return 0.3

        # More tools used = higher confidence (more thorough)
        tool_score = min(len(tools_used) / 3.0, 1.0)

        # More steps with successful observations = higher confidence
        success_score = sum(1 for s in trace if "error" not in s.observation.lower()) / max(len(trace), 1)

        return 0.5 * tool_score + 0.5 * success_score
