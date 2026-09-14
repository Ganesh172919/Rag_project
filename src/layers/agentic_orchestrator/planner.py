"""Multi-Step Planner — Plan and execute multi-step retrieval reasoning."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlanStep:
    """A single step in a multi-step plan."""
    step_id: int
    description: str
    tool: str
    input_query: str
    result: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class Plan:
    """A multi-step plan for answering a complex query."""
    query: str
    steps: List[PlanStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    status: str = "pending"


class MultiStepPlanner:
    """Plan multi-step retrieval for complex queries.

    Implements planning strategies:
    - Sequential: Execute steps one by one
    - Parallel: Execute independent steps in parallel
    - Adaptive: Adjust plan based on intermediate results
    """

    def __init__(self, llm_fn=None, max_steps: int = 5):
        self.llm_fn = llm_fn
        self.max_steps = max_steps

    def create_plan(self, query: str, routing_strategy: str) -> Plan:
        """Create a multi-step plan for answering a query.

        Args:
            query: The user's question
            routing_strategy: Strategy from the adaptive router

        Returns:
            Plan with ordered steps
        """
        if routing_strategy == "graph_global":
            return self._plan_graph_global(query)
        elif routing_strategy == "multi_step":
            return self._plan_multi_step(query)
        elif routing_strategy == "single_step":
            return self._plan_single_step(query)
        else:  # no_retrieval
            return self._plan_no_retrieval(query)

    def _plan_no_retrieval(self, query: str) -> Plan:
        """Plan for simple queries that don't need retrieval."""
        return Plan(
            query=query,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Generate answer directly from LLM knowledge",
                    tool="generate",
                    input_query=query,
                )
            ],
        )

    def _plan_single_step(self, query: str) -> Plan:
        """Plan for single-hop retrieval queries."""
        return Plan(
            query=query,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Retrieve relevant documents",
                    tool="retrieve",
                    input_query=query,
                ),
                PlanStep(
                    step_id=2,
                    description="Generate answer from retrieved context",
                    tool="generate",
                    input_query=query,
                ),
            ],
        )

    def _plan_multi_step(self, query: str) -> Plan:
        """Plan for multi-hop queries requiring multiple retrieval steps."""
        if self.llm_fn:
            return self._plan_multi_step_llm(query)
        return self._plan_multi_step_heuristic(query)

    def _plan_multi_step_heuristic(self, query: str) -> Plan:
        """Heuristic-based multi-step planning."""
        steps = [
            PlanStep(
                step_id=1,
                description="Decompose query into sub-queries",
                tool="decompose",
                input_query=query,
            ),
            PlanStep(
                step_id=2,
                description="Retrieve documents for sub-query 1",
                tool="retrieve",
                input_query=query,
            ),
            PlanStep(
                step_id=3,
                description="Retrieve documents for sub-query 2",
                tool="retrieve",
                input_query=query,
            ),
            PlanStep(
                step_id=4,
                description="Combine retrieved information",
                tool="generate",
                input_query=query,
            ),
            PlanStep(
                step_id=5,
                description="Verify and refine answer",
                tool="verify",
                input_query=query,
            ),
        ]
        return Plan(query=query, steps=steps)

    def _plan_multi_step_llm(self, query: str) -> Plan:
        """LLM-based multi-step planning."""
        prompt = f"""Create a step-by-step plan to answer this complex question.

Question: {query}

Available tools:
- retrieve: Search the local knowledge base
- graph_query: Query the knowledge graph
- web_search: Search the web
- decompose: Break query into sub-queries
- verify: Verify factual claims
- generate: Generate the final answer

Create a plan with 3-5 steps. For each step, specify:
1. What to do
2. Which tool to use
3. What query to pass to the tool

Plan:"""

        response = self.llm_fn(prompt).strip()
        steps = self._parse_plan_response(response, query)

        if not steps:
            return self._plan_multi_step_heuristic(query)

        return Plan(query=query, steps=steps)

    def _plan_graph_global(self, query: str) -> Plan:
        """Plan for global/corpus-level queries using knowledge graph."""
        steps = [
            PlanStep(
                step_id=1,
                description="Query knowledge graph for entities and relationships",
                tool="graph_query",
                input_query=query,
            ),
            PlanStep(
                step_id=2,
                description="Retrieve documents related to key entities",
                tool="retrieve",
                input_query=query,
            ),
            PlanStep(
                step_id=3,
                description="Generate comprehensive answer from graph + documents",
                tool="generate",
                input_query=query,
            ),
            PlanStep(
                step_id=4,
                description="Verify completeness and accuracy",
                tool="verify",
                input_query=query,
            ),
        ]
        return Plan(query=query, steps=steps)

    def _parse_plan_response(self, response: str, original_query: str) -> List[PlanStep]:
        """Parse LLM-generated plan into PlanStep objects."""
        steps = []
        lines = response.strip().split("\n")
        step_id = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for step indicators
            if any(line.startswith(p) for p in ["Step", "step", f"{step_id + 1}.", f"{step_id + 1})"]):
                step_id += 1
                # Try to extract tool name
                tool = "retrieve"  # default
                for tool_name in ["retrieve", "graph_query", "web_search", "decompose", "verify", "generate"]:
                    if tool_name in line.lower():
                        tool = tool_name
                        break

                steps.append(PlanStep(
                    step_id=step_id,
                    description=line,
                    tool=tool,
                    input_query=original_query,
                ))

            if step_id >= self.max_steps:
                break

        return steps
