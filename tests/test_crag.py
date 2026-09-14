"""Tests for Corrective Retrieval (CRAG)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from layers.corrective_retrieval.evaluator import RetrievalEvaluator
from layers.corrective_retrieval.decomposer import DocumentDecomposer, KnowledgeSnippet
from layers.corrective_retrieval.corrector import CorrectiveRetrieval


class TestRetrievalEvaluator:
    def setup_method(self):
        self.evaluator = RetrievalEvaluator()  # Uses heuristic fallback

    def test_relevant_docs_score_high(self):
        score = self.evaluator.score(
            "What is machine learning?",
            ["Machine learning is a subset of artificial intelligence that enables systems to learn."]
        )
        assert score > 0.3

    def test_irrelevant_docs_score_low(self):
        score = self.evaluator.score(
            "What is machine learning?",
            ["The weather today is sunny with clear skies."]
        )
        assert score < 0.5

    def test_empty_docs_score_zero(self):
        score = self.evaluator.score("test query", [])
        assert score == 0.0

    def test_per_document_scoring(self):
        scores = self.evaluator.score_per_document(
            "What is Python?",
            ["Python is a programming language.", "The cat sat on the mat."]
        )
        assert len(scores) == 2
        assert scores[0] > scores[1]


class TestDocumentDecomposer:
    def setup_method(self):
        self.decomposer = DocumentDecomposer()

    def test_decompose_sentences(self):
        doc = "First sentence here. Second sentence there. Third sentence everywhere."
        snippets = self.decomposer.decompose(doc, "doc1")
        assert len(snippets) >= 2

    def test_score_snippets(self):
        snippets = [
            KnowledgeSnippet(text="Python is a programming language.", source_doc="d1"),
            KnowledgeSnippet(text="The weather is nice today.", source_doc="d1"),
        ]
        scored = self.decomposer.score_snippets("What is Python?", snippets)
        assert scored[0].relevance_score > scored[1].relevance_score

    def test_filter_snippets(self):
        snippets = [
            KnowledgeSnippet(text="Python programming", source_doc="d1", relevance_score=0.8),
            KnowledgeSnippet(text="Weather forecast", source_doc="d1", relevance_score=0.1),
        ]
        filtered = self.decomposer.filter_snippets(snippets)
        assert len(filtered) == 1
        assert filtered[0].text == "Python programming"

    def test_decompose_recompose(self):
        docs = [
            "Python is a popular programming language used for many applications.",
            "The weather today is sunny and warm.",
        ]
        result = self.decomposer.decompose_recompose("What is Python?", docs)
        assert "Python" in result


class TestCorrectiveRetrieval:
    def setup_method(self):
        self.corrector = CorrectiveRetrieval()

    def test_process_with_empty_docs(self):
        # Should attempt web fallback or return empty
        result = self.corrector.process("test query", [], enable_web_fallback=False)
        assert result.action in ["INCORRECT", "INCORRECT_NO_FALLBACK"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
