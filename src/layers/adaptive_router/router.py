"""Adaptive Router — Route queries to appropriate retrieval strategies."""

from dataclasses import dataclass
from typing import Optional

from .classifier import QueryComplexityClassifier


@dataclass
class RoutingDecision:
    """The result of routing a query."""
    level: int  # 0-3
    strategy: str  # no_retrieval, single_step, multi_step, graph_global
    confidence: float
    query: str
    fallback_used: bool = False


STRATEGY_MAP = {
    0: "no_retrieval",
    1: "single_step",
    2: "multi_step",
    3: "graph_global",
}


class AdaptiveRouter:
    """Route queries to the appropriate RAG strategy based on complexity.

    Based on: Adaptive-RAG (Jeong et al., NAACL 2024)
    """

    def __init__(
        self,
        classifier_model: str = "microsoft/deberta-v3-base",
        confidence_threshold: float = 0.7,
        default_strategy: str = "multi_step",
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.default_strategy = default_strategy
        self.classifier = QueryComplexityClassifier(model_name=classifier_model, device=device)

    def route(self, query: str) -> RoutingDecision:
        """Route a query to the appropriate strategy.

        Args:
            query: The user's question

        Returns:
            RoutingDecision with level, strategy, and confidence
        """
        level, confidence = self.classifier.predict(query)

        # If confidence is low, use default safe strategy
        fallback_used = False
        if confidence < self.confidence_threshold:
            level = self._strategy_to_level(self.default_strategy)
            fallback_used = True

        strategy = STRATEGY_MAP.get(level, "multi_step")

        return RoutingDecision(
            level=level,
            strategy=strategy,
            confidence=confidence,
            query=query,
            fallback_used=fallback_used,
        )

    def route_batch(self, queries: list) -> list:
        """Route a batch of queries."""
        return [self.route(q) for q in queries]

    def _strategy_to_level(self, strategy: str) -> int:
        """Convert strategy name to level number."""
        for level, name in STRATEGY_MAP.items():
            if name == strategy:
                return level
        return 2  # Default to multi_step

    def save(self, path: str):
        """Save the router model."""
        self.classifier.save(path)

    def load(self, path: str):
        """Load the router model."""
        self.classifier.load(path)
