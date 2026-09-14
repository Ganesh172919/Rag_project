"""Hybrid Retrieval — Combines BM25 sparse and vector dense retrieval.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
retrieval methods, capturing both keyword matching and semantic similarity.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("aarag.layers.hybrid_retrieval")


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    content: str
    doc_id: str
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    source: str = "hybrid"


class BM25Index:
    """Simple BM25 index for sparse retrieval.

    Implements Okapi BM25 scoring for keyword-based retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.total_docs: int = 0
        self.index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]

    def add_documents(self, documents: List[Dict]):
        """Add documents to the BM25 index.

        Args:
            documents: List of dicts with 'content' and 'doc_id' keys
        """
        for doc in documents:
            doc_idx = len(self.documents)
            self.documents.append(doc)

            content = doc.get("content", "").lower()
            tokens = content.split()
            self.doc_lengths.append(len(tokens))

            # Count term frequencies
            term_freqs: Dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            # Update index
            for term, tf in term_freqs.items():
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append((doc_idx, tf))

            # Update document frequencies
            for term in term_freqs.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.total_docs = len(self.documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search the BM25 index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_index, score) tuples sorted by score
        """
        if not self.documents:
            return []

        query_tokens = query.lower().split()
        scores = [0.0] * self.total_docs

        for term in query_tokens:
            if term not in self.index:
                continue

            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, tf in self.index[term]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_length
                )
                scores[doc_idx] += idf * numerator / denominator

        # Sort by score
        scored = [(i, s) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def clear(self):
        """Clear the index."""
        self.documents.clear()
        self.doc_lengths.clear()
        self.index.clear()
        self.doc_freqs.clear()
        self.total_docs = 0
        self.avg_doc_length = 0.0


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    sparse (BM25) and dense (vector) retrieval.

    RRF Score(d) = Σ 1 / (k + rank_i(d))

    Where k=60 (standard parameter) and the sum is over retrieval methods.

    Example:
        >>> retriever = HybridRetriever(vector_store=vs)
        >>> retriever.add_documents(chunks)
        >>> results = retriever.search("What is ML?", top_k=5)
    """

    def __init__(
        self,
        vector_store=None,
        k: int = 60,
        alpha: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: VectorStore instance for dense retrieval
            k: RRF parameter (default: 60)
            alpha: Weight for dense retrieval (1-alpha for BM25)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.vector_store = vector_store
        self.k = k
        self.alpha = alpha
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self._documents: List[Dict] = []

        logger.info(
            f"HybridRetriever initialized (k={k}, alpha={alpha})"
        )

    def add_documents(self, documents: List[Dict]):
        """Add documents to both BM25 and dense indexes.

        Args:
            documents: List of dicts with 'content' and 'doc_id' keys
        """
        self._documents.extend(documents)
        self.bm25.add_documents(documents)

        # Add to vector store if available
        if self.vector_store is not None:
            try:
                from .knowledge_sources.document_loader import Chunk

                chunks = []
                for doc in documents:
                    chunk = Chunk(
                        content=doc.get("content", ""),
                        doc_id=doc.get("doc_id", ""),
                        metadata=doc.get("metadata", {}),
                    )
                    chunks.append(chunk)
                self.vector_store.add_documents(chunks)
            except Exception as e:
                logger.warning(f"Could not add to vector store: {e}")

        logger.info(f"Added {len(documents)} documents to hybrid index")

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
    ) -> List[HybridResult]:
        """Search using hybrid retrieval with RRF.

        Args:
            query: Search query
            top_k: Number of final results to return
            bm25_top_k: Number of BM25 results to retrieve (default: top_k * 3)
            dense_top_k: Number of dense results to retrieve (default: top_k * 3)

        Returns:
            List of HybridResult sorted by RRF score
        """
        if bm25_top_k is None:
            bm25_top_k = top_k * 3
        if dense_top_k is None:
            dense_top_k = top_k * 3

        # BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)
        bm25_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_results)}

        # Dense retrieval
        dense_ranks = {}
        if self.vector_store is not None:
            try:
                dense_results = self.vector_store.search(query, top_k=dense_top_k)
                for rank, result in enumerate(dense_results):
                    # Find matching document by content
                    for doc_idx, doc in enumerate(self._documents):
                        if doc.get("content", "") == result.content:
                            dense_ranks[doc_idx] = rank + 1
                            break
            except Exception as e:
                logger.warning(f"Dense retrieval failed: {e}")

        # Compute RRF scores
        rrf_scores: Dict[int, float] = {}

        all_doc_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        for doc_idx in all_doc_indices:
            score = 0.0

            # BM25 contribution
            if doc_idx in bm25_ranks:
                score += (1 - self.alpha) / (self.k + bm25_ranks[doc_idx])

            # Dense contribution
            if doc_idx in dense_ranks:
                score += self.alpha / (self.k + dense_ranks[doc_idx])

            rrf_scores[doc_idx] = score

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build results
        results = []
        for doc_idx in sorted_indices[:top_k]:
            doc = self._documents[doc_idx]
            results.append(HybridResult(
                content=doc.get("content", ""),
                doc_id=doc.get("doc_id", ""),
                score=rrf_scores[doc_idx],
                bm25_rank=bm25_ranks.get(doc_idx),
                dense_rank=dense_ranks.get(doc_idx),
                rrf_score=rrf_scores[doc_idx],
                source="hybrid",
            ))

        logger.debug(
            f"Hybrid search: {len(bm25_results)} BM25 + "
            f"{len(dense_ranks)} dense → {len(results)} results"
        )

        return results

    def clear(self):
        """Clear all indexes."""
        self.bm25.clear()
        self._documents.clear()
        logger.info("Hybrid index cleared")
