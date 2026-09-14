"""Tests for HybridRetriever."""

import pytest
from src.layers.hybrid_retrieval import HybridRetriever, BM25Index, HybridResult


class TestBM25Index:
    """Tests for BM25 index."""

    @pytest.fixture
    def bm25(self):
        """Create a BM25 index."""
        return BM25Index()

    @pytest.fixture
    def sample_docs(self):
        """Sample documents."""
        return [
            {"content": "machine learning is a branch of artificial intelligence", "doc_id": "1"},
            {"content": "deep learning uses neural networks", "doc_id": "2"},
            {"content": "python is a programming language", "doc_id": "3"},
            {"content": "natural language processing handles text", "doc_id": "4"},
        ]

    def test_add_documents(self, bm25, sample_docs):
        """Should add documents to index."""
        bm25.add_documents(sample_docs)
        assert bm25.total_docs == 4

    def test_search_returns_results(self, bm25, sample_docs):
        """Search should return results."""
        bm25.add_documents(sample_docs)
        results = bm25.search("machine learning")
        assert len(results) > 0

    def test_search_relevant_first(self, bm25, sample_docs):
        """Most relevant document should rank first."""
        bm25.add_documents(sample_docs)
        results = bm25.search("machine learning")
        # Doc 1 has "machine learning" - should be first
        assert results[0][0] == 0  # doc_idx 0

    def test_search_empty_index(self, bm25):
        """Search on empty index should return empty."""
        results = bm25.search("test")
        assert results == []

    def test_search_top_k(self, bm25, sample_docs):
        """top_k should limit results."""
        bm25.add_documents(sample_docs)
        results = bm25.search("learning", top_k=2)
        assert len(results) <= 2

    def test_clear(self, bm25, sample_docs):
        """Clear should empty the index."""
        bm25.add_documents(sample_docs)
        bm25.clear()
        assert bm25.total_docs == 0


class TestHybridRetriever:
    """Tests for hybrid retriever."""

    @pytest.fixture
    def retriever(self):
        """Create a hybrid retriever (no vector store)."""
        return HybridRetriever(vector_store=None, alpha=0.5)

    @pytest.fixture
    def sample_docs(self):
        """Sample documents."""
        return [
            {"content": "machine learning is a branch of artificial intelligence", "doc_id": "1"},
            {"content": "deep learning uses neural networks", "doc_id": "2"},
            {"content": "python is a programming language", "doc_id": "3"},
            {"content": "natural language processing handles text", "doc_id": "4"},
        ]

    def test_add_documents(self, retriever, sample_docs):
        """Should add documents to BM25 index."""
        retriever.add_documents(sample_docs)
        assert retriever.bm25.total_docs == 4

    def test_search_returns_results(self, retriever, sample_docs):
        """Search should return HybridResult objects."""
        retriever.add_documents(sample_docs)
        results = retriever.search("machine learning")
        assert len(results) > 0
        assert all(isinstance(r, HybridResult) for r in results)

    def test_search_sorted_by_score(self, retriever, sample_docs):
        """Results should be sorted by RRF score."""
        retriever.add_documents(sample_docs)
        results = retriever.search("machine learning")
        for i in range(len(results) - 1):
            assert results[i].rrf_score >= results[i + 1].rrf_score

    def test_search_top_k(self, retriever, sample_docs):
        """top_k should limit results."""
        retriever.add_documents(sample_docs)
        results = retriever.search("learning", top_k=2)
        assert len(results) <= 2

    def test_search_empty(self, retriever):
        """Search on empty retriever should return empty."""
        results = retriever.search("test")
        assert results == []

    def test_clear(self, retriever, sample_docs):
        """Clear should empty the retriever."""
        retriever.add_documents(sample_docs)
        retriever.clear()
        results = retriever.search("test")
        assert results == []
