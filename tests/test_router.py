"""Tests for the Adaptive Router."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from layers.adaptive_router.feature_extractor import FeatureExtractor
from layers.adaptive_router.classifier import _RuleBasedClassifier
from layers.adaptive_router.router import AdaptiveRouter


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_simple_query_features(self):
        features = self.extractor.extract("Who is Albert Einstein?")
        assert features["has_who"] == 1.0
        assert features["word_count"] == 4

    def test_multi_hop_features(self):
        features = self.extractor.extract("Compare the economic policies of the US and China")
        assert features["has_comparison"] == 1.0
        assert features["multi_hop_keyword_count"] >= 1

    def test_global_features(self):
        features = self.extractor.extract("Summarize all the main themes across the documents")
        assert features["global_keyword_count"] >= 1
        assert features["has_summary"] == 1.0

    def test_feature_names(self):
        names = self.extractor.get_feature_names()
        assert len(names) > 10
        assert "word_count" in names


class TestRuleBasedClassifier:
    def setup_method(self):
        self.classifier = _RuleBasedClassifier()

    def test_simple_factual(self):
        level, confidence = self.classifier.predict("Who is the president of France?")
        assert level == 0
        assert confidence > 0.5

    def test_multi_hop(self):
        level, confidence = self.classifier.predict("Compare Python and Java programming languages")
        assert level == 2

    def test_global(self):
        level, confidence = self.classifier.predict("Summarize all the main topics in the document")
        assert level == 3

    def test_single_hop(self):
        level, confidence = self.classifier.predict("What is the capital of France?")
        assert level in [0, 1]


class TestAdaptiveRouter:
    def setup_method(self):
        self.router = AdaptiveRouter(confidence_threshold=0.5)

    def test_route_returns_decision(self):
        decision = self.router.route("What is machine learning?")
        assert decision.level in [0, 1, 2, 3]
        assert decision.strategy in ["no_retrieval", "single_step", "multi_step", "graph_global"]
        assert 0 <= decision.confidence <= 1

    def test_route_batch(self):
        queries = ["Who is Einstein?", "Compare A and B", "Summarize everything"]
        decisions = self.router.route_batch(queries)
        assert len(decisions) == 3
        for d in decisions:
            assert d.strategy in ["no_retrieval", "single_step", "multi_step", "graph_global"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
