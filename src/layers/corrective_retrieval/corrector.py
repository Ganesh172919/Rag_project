"""Corrective Retrieval — Main CRAG integration module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .evaluator import RetrievalEvaluator
from .decomposer import DocumentDecomposer
from .web_fallback import WebFallback
from ..knowledge_sources.vector_store import RetrievalResult


@dataclass
class CorrectiveResult:
    """Result from the corrective retrieval pipeline."""
    context: str
    action: str  # CORRECT, INCORRECT, AMBIGUOUS
    confidence: float
    sources: List[str]
    web_used: bool = False


class CorrectiveRetrieval:
    """Corrective Retrieval Augmented Generation (CRAG) pipeline.

    Based on: CRAG (Yan et al., ICML 2024)

    Evaluates retrieval quality and takes corrective action:
    - CORRECT (confidence > 0.8): Use retrieved docs as-is
    - INCORRECT (confidence < 0.3): Fall back to web search
    - AMBIGUOUS (0.3-0.8): Combine local + web, decompose-recompose
    """

    def __init__(
        self,
        evaluator_model: str = "microsoft/deberta-v3-small",
        web_provider: str = "duckduckgo",
        correct_threshold: float = 0.8,
        incorrect_threshold: float = 0.3,
        max_web_results: int = 5,
        device: str = "auto",
    ):
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold

        self.evaluator = RetrievalEvaluator(model_name=evaluator_model, device=device)
        self.decomposer = DocumentDecomposer()
        self.web_fallback = WebFallback(provider=web_provider, max_results=max_web_results)

    def process(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        enable_web_fallback: bool = True,
    ) -> CorrectiveResult:
        """Process retrieved documents through the CRAG pipeline.

        Args:
            query: The user's question
            retrieved_docs: Documents from the vector store
            enable_web_fallback: Whether to use web search as fallback

        Returns:
            CorrectiveResult with filtered/expanded context
        """
        # Extract document texts
        doc_texts = [doc.content for doc in retrieved_docs]
        sources = [doc.metadata.get("source", "local") for doc in retrieved_docs]

        if not doc_texts:
            # No documents retrieved — go directly to web
            if enable_web_fallback:
                return self._handle_incorrect(query, sources)
            return CorrectiveResult(
                context="",
                action="INCORRECT",
                confidence=0.0,
                sources=[],
            )

        # Evaluate retrieval quality
        confidence = self.evaluator.score(query, doc_texts)

        if confidence > self.correct_threshold:
            # CORRECT: Use decompose-recompose to extract key information
            context = self.decomposer.keypoint_extraction(doc_texts)
            return CorrectiveResult(
                context=context,
                action="CORRECT",
                confidence=confidence,
                sources=sources,
            )

        elif confidence < self.incorrect_threshold:
            # INCORRECT: Fall back to web search
            if enable_web_fallback:
                return self._handle_incorrect(query, sources)
            # If web fallback disabled, use what we have with filtering
            context = self.decomposer.decompose_recompose(query, doc_texts)
            return CorrectiveResult(
                context=context,
                action="INCORRECT_NO_FALLBACK",
                confidence=confidence,
                sources=sources,
            )

        else:
            # AMBIGUOUS: Combine local + web, decompose-recompose
            return self._handle_ambiguous(query, doc_texts, sources, enable_web_fallback)

    def _handle_incorrect(self, query: str, original_sources: List[str]) -> CorrectiveResult:
        """Handle case where retrieval is incorrect — use web search."""
        web_texts = self.web_fallback.search(query)

        if web_texts:
            # Extract key points from web results
            context = self.decomposer.keypoint_extraction(web_texts)
            return CorrectiveResult(
                context=context,
                action="INCORRECT",
                confidence=0.1,
                sources=["web_search"],
                web_used=True,
            )
        else:
            # Web search also failed
            return CorrectiveResult(
                context="",
                action="INCORRECT",
                confidence=0.0,
                sources=[],
                web_used=True,
            )

    def _handle_ambiguous(
        self,
        query: str,
        doc_texts: List[str],
        sources: List[str],
        enable_web_fallback: bool,
    ) -> CorrectiveResult:
        """Handle ambiguous retrieval — combine local + web results."""
        all_texts = list(doc_texts)
        all_sources = list(sources)

        if enable_web_fallback:
            web_texts = self.web_fallback.search(query)
            all_texts.extend(web_texts)
            all_sources.extend(["web_search"] * len(web_texts))

        # Decompose-recompose over combined results
        context = self.decomposer.decompose_recompose(query, all_texts)

        return CorrectiveResult(
            context=context,
            action="AMBIGUOUS",
            confidence=0.5,
            sources=all_sources,
            web_used=enable_web_fallback,
        )
