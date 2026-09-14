"""Tests for CrossEncoderReranker."""

import pytest
from src.layers.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Tests for the cross-encoder reranker."""

    @pytest.fixture
    def reranker(self):
        """Create a reranker with fallback (no model download)."""
        return CrossEncoderReranker()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"content": "Machine learning is a subset of artificial intelligence.", "doc_id": "1", "score": 0.8},
            {"content": "Python is a popular programming language.", "doc_id": "2", "score": 0.6},
            {"content": "Deep learning uses neural networks with multiple layers.", "doc_id": "3", "score": 0.7},
            {"content": "The weather today is sunny and warm.", "doc_id": "4", "score": 0.3},
            {"content": "Natural language processing deals with text data.", "doc_id": "5", "score": 0.65},
        ]

    def test_rerank_returns_sorted_results(self, reranker, sample_documents):
        """Reranked results should be sorted by combined score."""
        results = reranker.rerank("What is machine learning?", sample_documents)
        assert len(results) == len(sample_documents)
        for i in range(len(results) - 1):
            assert results[i]["combined_score"] >= results[i + 1]["combined_score"]

    def test_rerank_preserves_documents(self, reranker, sample_documents):
        """Reranking should preserve all documents."""
        results = reranker.rerank("test query", sample_documents)
        assert len(results) == len(sample_documents)
        result_ids = {r["doc_id"] for r in results}
        original_ids = {d["doc_id"] for d in sample_documents}
        assert result_ids == original_ids

    def test_rerank_with_top_k(self, reranker, sample_documents):
        """top_k should limit the number of results."""
        results = reranker.rerank("machine learning", sample_documents, top_k=3)
        assert len(results) == 3

    def test_rerank_empty_documents(self, reranker):
        """Reranking empty documents should return empty list."""
        results = reranker.rerank("test", [])
        assert results == []

    def test_rerank_adds_scores(self, reranker, sample_documents):
        """Reranking should add reranker_score and combined_score."""
        results = reranker.rerank("test", sample_documents)
        for doc in results:
            assert "reranker_score" in doc
            assert "combined_score" in doc

    def test_fallback_rerank_keyword_overlap(self, reranker):
        """Fallback reranker should rank keyword-matching docs higher."""
        docs = [
            {"content": "completely unrelated content", "doc_id": "1"},
            {"content": "machine learning is great", "doc_id": "2"},
            {"content": "something else entirely", "doc_id": "3"},
        ]
        results = reranker.rerank("machine learning", docs)
        # Doc with "machine learning" should rank first
        assert results[0]["doc_id"] == "2"

    def test_save_load_config(self, reranker, tmp_path):
        """Should be able to save and load configuration."""
        reranker.save(str(tmp_path / "reranker"))
        assert (tmp_path / "reranker" / "config.json").exists()

        new_reranker = CrossEncoderReranker()
        new_reranker.load(str(tmp_path / "reranker"))
        assert new_reranker.model_name == reranker.model_name
