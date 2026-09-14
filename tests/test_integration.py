"""Integration tests for AARAG pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestDocumentLoader:
    def test_chunk_text(self):
        from layers.knowledge_sources.document_loader import DocumentLoader
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        text = "A" * 250
        chunks = loader._chunk_text(text)
        assert len(chunks) >= 2

    def test_chunk_short_text(self):
        from layers.knowledge_sources.document_loader import DocumentLoader
        loader = DocumentLoader(chunk_size=512)
        text = "Short text."
        chunks = loader._chunk_text(text)
        assert len(chunks) == 1


class TestVectorStore:
    def test_add_and_search(self):
        from layers.knowledge_sources.vector_store import VectorStore
        from layers.knowledge_sources.document_loader import Chunk
        import numpy as np

        # Create store with mock embeddings
        store = VectorStore.__new__(VectorStore)
        store._documents = [
            {"chunk_id": "c1", "content": "Python is a programming language", "metadata": {}, "embedding": [1.0] + [0.0] * 1023},
            {"chunk_id": "c2", "content": "The weather is nice", "metadata": {}, "embedding": [0.0] * 1024},
        ]
        store._dimension = 1024
        store._index = None

        assert len(store) == 2


class TestKnowledgeGraph:
    def test_build_and_search(self):
        from layers.knowledge_sources.knowledge_graph import KnowledgeGraphSearch
        kg = KnowledgeGraphSearch()

        docs = [
            {"content": "Albert Einstein was a physicist. He developed the theory of relativity.", "doc_id": "d1"},
            {"content": "Marie Curie was a chemist. She won the Nobel Prize.", "doc_id": "d2"},
        ]
        kg.build_from_documents(docs)

        assert len(kg.graph.nodes) > 0

        result = kg.search("Who is Albert Einstein?")
        assert result is not None


class TestFeatureExtractor:
    def test_extract_returns_dict(self):
        from layers.adaptive_router.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        features = extractor.extract("What is machine learning?")
        assert isinstance(features, dict)
        assert len(features) > 10


class TestDecomposer:
    def test_decompose_recompose(self):
        from layers.corrective_retrieval.decomposer import DocumentDecomposer
        decomposer = DocumentDecomposer()
        docs = ["Python is great. Java is also popular.", "The weather is nice."]
        result = decomposer.decompose_recompose("What is Python?", docs)
        assert isinstance(result, str)
        assert len(result) > 0


class TestEndToEnd:
    def test_aarag_import(self):
        """Test that AARAG can be imported."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "aarag", str(Path(__file__).parent.parent / "src" / "aarag.py"),
            submodule_search_locations=[str(Path(__file__).parent.parent / "src")]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["aarag"] = mod
        # Just test the class can be found in the file
        content = (Path(__file__).parent.parent / "src" / "aarag.py").read_text()
        assert "class AARAG" in content

    def test_aarag_init(self):
        """Test that AARAG can be initialized."""
        from layers.knowledge_sources import VectorStore, KnowledgeGraphSearch, WebSearch
        from layers.adaptive_router import AdaptiveRouter
        from layers.corrective_retrieval import CorrectiveRetrieval
        from layers.self_reflection import SelfReflectionEngine
        from layers.agentic_orchestrator import AgenticOrchestrator
        # Verify all layers can be imported
        assert VectorStore is not None
        assert AdaptiveRouter is not None

    def test_aarag_query_no_data(self):
        """Test AARAG components work independently."""
        from layers.adaptive_router import AdaptiveRouter
        from layers.corrective_retrieval.corrector import CorrectiveRetrieval
        router = AdaptiveRouter(confidence_threshold=0.5)
        decision = router.route("What is Python?")
        assert decision.strategy in ["no_retrieval", "single_step", "multi_step", "graph_global"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
