"""Baselines — Implement baseline systems for comparison."""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class BaselineResponse:
    """Response from a baseline system."""
    answer: str
    system_name: str
    latency: float = 0.0


class VanillaLLM:
    """Baseline: LLM without any retrieval."""

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn or self._default_fn
        self.name = "Vanilla LLM"

    def answer(self, query: str) -> BaselineResponse:
        import time
        start = time.time()
        response = self.llm_fn(f"Answer this question: {query}")
        elapsed = time.time() - start
        return BaselineResponse(answer=response, system_name=self.name, latency=elapsed)

    def _default_fn(self, prompt: str) -> str:
        return f"I don't have enough information to answer this question accurately."


class StandardRAG:
    """Baseline: Simple retrieve-then-read."""

    def __init__(self, vector_store=None, llm_fn=None):
        self.vector_store = vector_store
        self.llm_fn = llm_fn or self._default_fn
        self.name = "Standard RAG"

    def answer(self, query: str, top_k: int = 5) -> BaselineResponse:
        import time
        start = time.time()

        # Retrieve
        if self.vector_store:
            docs = self.vector_store.search(query, top_k=top_k)
            context = "\n\n".join(d.content[:300] for d in docs)
        else:
            context = ""

        # Generate
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
        response = self.llm_fn(prompt)

        elapsed = time.time() - start
        return BaselineResponse(answer=response, system_name=self.name, latency=elapsed)

    def _default_fn(self, prompt: str) -> str:
        if "Context:" in prompt:
            ctx = prompt.split("Context:")[1].split("\n\n")[0][:300]
            return f"Based on the context: {ctx}"
        return "No context available."


class SelfRAGBaseline:
    """Baseline: Self-RAG without corrective retrieval or routing."""

    def __init__(self, vector_store=None, self_reflection=None, llm_fn=None):
        self.vector_store = vector_store
        self.self_reflection = self_reflection
        self.llm_fn = llm_fn or self._default_fn
        self.name = "Self-RAG"

    def answer(self, query: str, top_k: int = 5) -> BaselineResponse:
        import time
        start = time.time()

        # Retrieve
        if self.vector_store:
            docs = self.vector_store.search(query, top_k=top_k)
            context = "\n\n".join(d.content[:300] for d in docs)
        else:
            context = ""

        # Generate
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.llm_fn(prompt)

        # Self-reflect
        if self.self_reflection:
            result = self.self_reflection.reflect_and_generate(query, context)
            if result.revised:
                answer = result.answer

        elapsed = time.time() - start
        return BaselineResponse(answer=answer, system_name=self.name, latency=elapsed)

    def _default_fn(self, prompt: str) -> str:
        if "Context:" in prompt:
            ctx = prompt.split("Context:")[1].split("\n\n")[0][:300]
            return f"Based on the context: {ctx}"
        return "No context available."


class CRAGBaseline:
    """Baseline: CRAG without self-reflection or routing."""

    def __init__(self, vector_store=None, crag=None, llm_fn=None):
        self.vector_store = vector_store
        self.crag = crag
        self.llm_fn = llm_fn or self._default_fn
        self.name = "CRAG"

    def answer(self, query: str, top_k: int = 5) -> BaselineResponse:
        import time
        start = time.time()

        # Retrieve
        if self.vector_store:
            docs = self.vector_store.search(query, top_k=top_k)
        else:
            docs = []

        # Corrective retrieval
        if self.crag:
            result = self.crag.process(query, docs)
            context = result.context
        else:
            context = "\n\n".join(d.content[:300] for d in docs)

        # Generate
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.llm_fn(prompt)

        elapsed = time.time() - start
        return BaselineResponse(answer=answer, system_name=self.name, latency=elapsed)

    def _default_fn(self, prompt: str) -> str:
        if "Context:" in prompt:
            ctx = prompt.split("Context:")[1].split("\n\n")[0][:300]
            return f"Based on the context: {ctx}"
        return "No context available."


class AdaptiveRAGBaseline:
    """Baseline: Adaptive-RAG without CRAG or self-reflection."""

    def __init__(self, vector_store=None, router=None, llm_fn=None):
        self.vector_store = vector_store
        self.router = router
        self.llm_fn = llm_fn or self._default_fn
        self.name = "Adaptive-RAG"

    def answer(self, query: str) -> BaselineResponse:
        import time
        start = time.time()

        # Route
        if self.router:
            routing = self.router.route(query)
            strategy = routing.strategy
        else:
            strategy = "single_step"

        # Retrieve based on strategy
        if strategy != "no_retrieval" and self.vector_store:
            docs = self.vector_store.search(query, top_k=5)
            context = "\n\n".join(d.content[:300] for d in docs)
        else:
            context = ""

        # Generate
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        answer = self.llm_fn(prompt)

        elapsed = time.time() - start
        return BaselineResponse(answer=answer, system_name=self.name, latency=elapsed)

    def _default_fn(self, prompt: str) -> str:
        if "Context:" in prompt:
            ctx = prompt.split("Context:")[1].split("\n\n")[0][:300]
            return f"Based on the context: {ctx}"
        return "No context available."


def get_all_baselines(
    vector_store=None,
    router=None,
    crag=None,
    self_reflection=None,
    llm_fn=None,
) -> Dict[str, object]:
    """Get all baseline systems."""
    return {
        "vanilla_llm": VanillaLLM(llm_fn=llm_fn),
        "standard_rag": StandardRAG(vector_store=vector_store, llm_fn=llm_fn),
        "self_rag": SelfRAGBaseline(vector_store=vector_store, self_reflection=self_reflection, llm_fn=llm_fn),
        "crag": CRAGBaseline(vector_store=vector_store, crag=crag, llm_fn=llm_fn),
        "adaptive_rag": AdaptiveRAGBaseline(vector_store=vector_store, router=router, llm_fn=llm_fn),
    }
