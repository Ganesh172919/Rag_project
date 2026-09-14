"""Vector Store — FAISS/ChromaDB-based dense retrieval."""

import os
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .document_loader import Chunk


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""
    content: str
    score: float
    chunk_id: str
    metadata: dict


class VectorStore:
    """Dense vector store using FAISS or ChromaDB for similarity search."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        backend: str = "faiss",
        persist_dir: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        self.backend = backend
        self.persist_dir = persist_dir
        self._embedder = None
        self._index = None
        self._documents: List[dict] = []
        self._dimension: int = 1024

    @property
    def embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            self._dimension = self._embedder.get_sentence_embedding_dimension()
        return self._embedder

    def add_documents(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Add chunks to the vector store. Returns number added."""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._encode_batch(texts, batch_size)

        for chunk, embedding in zip(chunks, embeddings):
            self._documents.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": embedding.tolist(),
            })

        self._build_index()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for most similar chunks to the query."""
        if not self._documents:
            return []

        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]

        if self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k)
        else:
            return self._search_chroma(query_embedding, top_k)

    def save(self, path: Optional[str] = None):
        """Persist the vector store to disk."""
        save_path = Path(path or self.persist_dir or "./data/vector_store")
        save_path.mkdir(parents=True, exist_ok=True)

        # Save documents
        docs_path = save_path / "documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f)

        # Save FAISS index
        if self._index is not None and self.backend == "faiss":
            import faiss
            faiss.write_index(self._index, str(save_path / "faiss.index"))

    def load(self, path: Optional[str] = None):
        """Load a persisted vector store from disk."""
        load_path = Path(path or self.persist_dir or "./data/vector_store")

        docs_path = load_path / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                self._documents = json.load(f)

        index_path = load_path / "faiss.index"
        if index_path.exists() and self.backend == "faiss":
            import faiss
            self._index = faiss.read_index(str(index_path))

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def _build_index(self):
        """Build or rebuild the search index."""
        if not self._documents:
            return

        embeddings = np.array([doc["embedding"] for doc in self._documents], dtype=np.float32)

        if self.backend == "faiss":
            import faiss
            if len(self._documents) < 1000:
                # Use flat index for small collections
                self._index = faiss.IndexFlatIP(self._dimension)
            else:
                # Use IVF index for larger collections
                nlist = min(int(np.sqrt(len(self._documents))), 256)
                quantizer = faiss.IndexFlatIP(self._dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self._dimension, nlist)
                self._index.train(embeddings)

            self._index.add(embeddings)

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """Search using FAISS index."""
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._documents):
                doc = self._documents[idx]
                results.append(RetrievalResult(
                    content=doc["content"],
                    score=float(score),
                    chunk_id=doc["chunk_id"],
                    metadata=doc.get("metadata", {}),
                ))
        return results

    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """Search using ChromaDB."""
        try:
            import chromadb
            client = chromadb.Client()
            collection = client.get_or_create_collection("aarag")
            # ChromaDB search would go here
            return []
        except Exception:
            return []

    def __len__(self) -> int:
        return len(self._documents)
