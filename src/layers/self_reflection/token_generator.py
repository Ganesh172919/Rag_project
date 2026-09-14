"""Reflection Token Generator — Generate reflection tokens for self-critique."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ReflectionTokens:
    """Container for reflection token outputs."""
    retrieve: str  # [Retrieve] or [No Retrieve]
    relevance: str  # [Relevant] or [Irrelevant]
    support: str  # [Fully Supported], [Partially Supported], or [No Support]
    utility: str  # [Very Useful], [Useful], or [Not Useful]

    @property
    def is_retrieval_needed(self) -> bool:
        return self.retrieve == "[Retrieve]"

    @property
    def is_relevant(self) -> bool:
        return self.relevance == "[Relevant]"

    @property
    def is_supported(self) -> bool:
        return self.support in ["[Fully Supported]", "[Partially Supported]"]

    @property
    def is_useful(self) -> bool:
        return self.utility in ["[Very Useful]", "[Useful]"]

    @property
    def needs_revision(self) -> bool:
        return not (self.is_relevant and self.is_supported and self.is_useful)


class ReflectionTokenGenerator:
    """Generate reflection tokens for self-critique.

    Based on: Self-RAG (Asai et al., ICLR 2024)

    Reflection tokens:
    - [Retrieve] / [No Retrieve]: Should we retrieve?
    - [Relevant] / [Irrelevant]: Is retrieved context relevant?
    - [Fully Supported] / [Partially Supported] / [No Support]: Is answer supported by context?
    - [Very Useful] / [Useful] / [Not Useful]: Is the answer useful?
    """

    def __init__(self, llm_fn=None):
        """
        Args:
            llm_fn: Function that takes a prompt and returns generated text.
                    If None, uses heuristic-based token generation.
        """
        self.llm_fn = llm_fn

    def generate_retrieve_token(self, query: str) -> str:
        """Decide whether retrieval is needed for this query."""
        if self.llm_fn:
            prompt = f"""Given the following question, decide if external knowledge retrieval is needed.

Question: {query}

If the question requires factual knowledge, recent information, or specific details not commonly known, respond with [Retrieve].
If the question is simple, opinion-based, or can be answered from common knowledge, respond with [No Retrieve].

Answer:"""
            response = self.llm_fn(prompt).strip()
            if "[Retrieve]" in response:
                return "[Retrieve]"
            return "[No Retrieve]"
        return self._heuristic_retrieve(query)

    def generate_relevance_token(self, query: str, context: str) -> str:
        """Evaluate if the retrieved context is relevant to the query."""
        if self.llm_fn:
            prompt = f"""Evaluate if the following context is relevant to answering the question.

Question: {query}
Context: {context[:1000]}

If the context contains information useful for answering the question, respond with [Relevant].
If the context does not contain useful information, respond with [Irrelevant].

Answer:"""
            response = self.llm_fn(prompt).strip()
            if "[Relevant]" in response:
                return "[Relevant]"
            return "[Irrelevant]"
        return self._heuristic_relevance(query, context)

    def generate_support_token(self, answer: str, context: str) -> str:
        """Evaluate if the answer is supported by the context."""
        if self.llm_fn:
            prompt = f"""Evaluate if the following answer is supported by the given context.

Answer: {answer}
Context: {context[:1000]}

Classify the support level:
- [Fully Supported]: Every claim in the answer is directly stated or implied in the context.
- [Partially Supported]: Some claims are supported, but some are not found in the context.
- [No Support]: The answer's claims cannot be found in the context.

Answer:"""
            response = self.llm_fn(prompt).strip()
            if "[Fully Supported]" in response:
                return "[Fully Supported]"
            elif "[Partially Supported]" in response:
                return "[Partially Supported]"
            return "[No Support]"
        return self._heuristic_support(answer, context)

    def generate_utility_token(self, query: str, answer: str) -> str:
        """Evaluate if the answer is useful for the query."""
        if self.llm_fn:
            prompt = f"""Evaluate if the following answer is useful for the question.

Question: {query}
Answer: {answer}

- [Very Useful]: The answer directly and completely addresses the question.
- [Useful]: The answer partially addresses the question.
- [Not Useful]: The answer does not address the question.

Answer:"""
            response = self.llm_fn(prompt).strip()
            if "[Very Useful]" in response:
                return "[Very Useful]"
            elif "[Useful]" in response:
                return "[Useful]"
            return "[Not Useful]"
        return self._heuristic_utility(query, answer)

    def generate_all(
        self,
        query: str,
        context: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> ReflectionTokens:
        """Generate all reflection tokens."""
        retrieve = self.generate_retrieve_token(query)

        relevance = "[Relevant]"
        support = "[Fully Supported]"
        utility = "[Very Useful]"

        if context:
            relevance = self.generate_relevance_token(query, context)
        if answer and context:
            support = self.generate_support_token(answer, context)
        if answer:
            utility = self.generate_utility_token(query, answer)

        return ReflectionTokens(
            retrieve=retrieve,
            relevance=relevance,
            support=support,
            utility=utility,
        )

    # Heuristic fallbacks

    def _heuristic_retrieve(self, query: str) -> str:
        """Simple heuristic: retrieve for factual questions."""
        factual_starters = ["who", "what", "when", "where", "how many", "how much",
                          "which", "name", "list"]
        query_lower = query.lower().strip()
        if any(query_lower.startswith(s) for s in factual_starters):
            return "[Retrieve]"
        if len(query.split()) > 10:
            return "[Retrieve]"
        return "[No Retrieve]"

    def _heuristic_relevance(self, query: str, context: str) -> str:
        """Simple heuristic: keyword overlap."""
        import string
        query_words = set(query.lower().translate(str.maketrans("", "", string.punctuation)).split()) - {"the", "a", "an", "is", "are", "what", "who", "when", "where", "how", "of", "in", "to", "be"}
        context_lower = context.lower().translate(str.maketrans("", "", string.punctuation))
        # Check both word overlap and substring presence
        overlap = sum(1 for w in query_words if w in context_lower)
        if overlap >= max(1, len(query_words) * 0.25):
            return "[Relevant]"
        return "[Irrelevant]"

    def _heuristic_support(self, answer: str, context: str) -> str:
        """Simple heuristic: answer words found in context."""
        answer_words = set(answer.lower().split()) - {"the", "a", "an", "is", "are", "was", "were"}
        context_lower = context.lower()
        found = sum(1 for w in answer_words if w in context_lower)
        ratio = found / max(len(answer_words), 1)
        if ratio > 0.6:
            return "[Fully Supported]"
        elif ratio > 0.3:
            return "[Partially Supported]"
        return "[No Support]"

    def _heuristic_utility(self, query: str, answer: str) -> str:
        """Simple heuristic: answer length and relevance."""
        if len(answer.split()) < 3:
            return "[Not Useful]"
        query_words = set(query.lower().split()) - {"the", "a", "an", "is", "are", "what", "who"}
        answer_lower = answer.lower()
        overlap = sum(1 for w in query_words if w in answer_lower)
        if overlap >= max(2, len(query_words) * 0.4):
            return "[Very Useful]"
        elif overlap >= 1:
            return "[Useful]"
        return "[Not Useful]"
