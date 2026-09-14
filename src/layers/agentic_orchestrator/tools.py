"""Tool Registry — Define and manage tools available to the agent."""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List


@dataclass
class Tool:
    """A tool available to the agent."""
    name: str
    description: str
    fn: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return self.fn(**kwargs)


class ToolRegistry:
    """Registry of tools available to the agentic orchestrator."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the agent prompt."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}")
        return tool.execute(**kwargs)


def create_default_tools(
    vector_store=None,
    knowledge_graph=None,
    web_search=None,
    crag=None,
    self_reflection=None,
) -> ToolRegistry:
    """Create the default tool registry for AARAG.

    Args:
        vector_store: VectorStore instance
        knowledge_graph: KnowledgeGraphSearch instance
        web_search: WebSearch instance
        crag: CorrectiveRetrieval instance
        self_reflection: SelfReflectionEngine instance

        Returns:
            ToolRegistry with default tools registered
    """
    registry = ToolRegistry()

    # Tool: retrieve — Search local knowledge base
    def retrieve(query: str, top_k: int = 5) -> str:
        if vector_store is None:
            return "Vector store not available."
        results = vector_store.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found in the knowledge base."
        texts = [f"[Score: {r.score:.2f}] {r.content[:300]}" for r in results]
        return "\n\n".join(texts)

    registry.register(Tool(
        name="retrieve",
        description="Search the local knowledge base for relevant documents.",
        fn=retrieve,
        parameters={"query": "str", "top_k": "int"},
    ))

    # Tool: graph_query — Query knowledge graph
    def graph_query(query: str) -> str:
        if knowledge_graph is None:
            return "Knowledge graph not available."
        result = knowledge_graph.search(query)
        if not result.entities:
            return "No relevant entities found in the knowledge graph."

        parts = []
        if result.entities:
            entities_str = ", ".join(e.name for e in result.entities[:5])
            parts.append(f"Related entities: {entities_str}")
        if result.relations:
            rels = [f"{r.source} --[{r.relation}]--> {r.target}" for r in result.relations[:5]]
            parts.append(f"Relations: {'; '.join(rels)}")
        if result.community_summaries:
            parts.append(f"Community context: {result.community_summaries[0]}")
        return "\n".join(parts)

    registry.register(Tool(
        name="graph_query",
        description="Query the knowledge graph for entity relationships and community context.",
        fn=graph_query,
        parameters={"query": "str"},
    ))

    # Tool: web_search — Search the web
    def web_search_tool(query: str) -> str:
        if web_search is None:
            return "Web search not available."
        results = web_search.search(query)
        if not results:
            return "No web results found."
        texts = [f"{r.title}: {r.snippet}" for r in results]
        return "\n\n".join(texts)

    registry.register(Tool(
        name="web_search",
        description="Search the web for up-to-date information.",
        fn=web_search_tool,
        parameters={"query": "str"},
    ))

    # Tool: self_reflect — Reflect on generation quality
    def self_reflect(query: str, context: str, answer: str) -> str:
        if self_reflection is None:
            return "Self-reflection not available."
        result = self_reflection.reflect_and_generate(query, context)
        tokens = result.tokens
        return (
            f"Reflection: Relevance={tokens.relevance}, "
            f"Support={tokens.support}, Utility={tokens.utility}, "
            f"Confidence={result.final_confidence:.2f}"
        )

    registry.register(Tool(
        name="self_reflect",
        description="Reflect on the quality of a generated answer.",
        fn=self_reflect,
        parameters={"query": "str", "context": "str", "answer": "str"},
    ))

    # Tool: decompose — Break complex query into sub-queries
    def decompose(query: str) -> str:
        # Simple decomposition heuristic
        parts = []
        if " and " in query.lower():
            segments = query.split(" and ")
            for i, seg in enumerate(segments):
                parts.append(f"Sub-query {i+1}: {seg.strip()}")
        elif "compare" in query.lower() or "difference" in query.lower():
            parts.append(f"Aspect 1: What is mentioned about the first entity in: {query}")
            parts.append(f"Aspect 2: What is mentioned about the second entity in: {query}")
        else:
            parts.append(f"Single query: {query}")
        return "\n".join(parts)

    registry.register(Tool(
        name="decompose",
        description="Break a complex query into simpler sub-queries.",
        fn=decompose,
        parameters={"query": "str"},
    ))

    # Tool: verify — Verify factual claims
    def verify(answer: str, context: str) -> str:
        if self_reflection is None:
            return "Verification not available."
        verifier = self_reflection.claim_verifier
        result = verifier.verify_answer(answer, context)
        return (
            f"Verification: {result['verified_count']}/{result['total_claims']} claims verified, "
            f"{result['contradicted_count']} contradicted, "
            f"Overall confidence: {result['overall_confidence']:.2f}"
        )

    registry.register(Tool(
        name="verify",
        description="Verify factual claims in an answer against the context.",
        fn=verify,
        parameters={"answer": "str", "context": "str"},
    ))

    return registry
