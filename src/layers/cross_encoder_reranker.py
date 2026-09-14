"""Cross-Encoder Reranker — Improves retrieval precision using cross-encoder scoring.

Uses a cross-encoder model to rerank retrieved documents, providing more
accurate ranking than bi-encoder similarity alone.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger("aarag.layers.cross_encoder")


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval precision.

    Reranks a list of retrieved documents using a cross-encoder model
    that computes query-document relevance scores.

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> results = reranker.rerank("What is ML?", documents)
        >>> for doc in results:
        ...     print(f"{doc.score:.3f}: {doc.content[:50]}")
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            max_length: Maximum sequence length for tokenizer
            batch_size: Batch size for inference
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._initialized = False

        logger.info(f"CrossEncoderReranker initialized (model={model_name})")

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=device,
            )
            self._initialized = True
            logger.info(f"Cross-encoder model loaded on {device}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback keyword-overlap reranker."
            )
            self._initialized = True
        except Exception as e:
            logger.warning(f"Could not load cross-encoder: {e}. Using fallback.")
            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of dicts with 'content' key and optional 'score'
            top_k: Return only top-k results (None = return all)

        Returns:
            List of dicts sorted by relevance score (highest first)
        """
        if not documents:
            return []

        self._load_model()

        if self._model is not None:
            return self._rerank_with_model(query, documents, top_k)
        else:
            return self._rerank_fallback(query, documents, top_k)

    def _rerank_with_model(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int],
    ) -> List[dict]:
        """Rerank using the cross-encoder model."""
        # Prepare query-document pairs
        pairs = [(query, doc.get("content", "")) for doc in documents]

        # Compute scores
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = dict(doc)
            scored_doc["reranker_score"] = float(score)
            # Combine with original score if present
            original_score = doc.get("score", 0.0)
            scored_doc["combined_score"] = 0.7 * float(score) + 0.3 * original_score
            scored_docs.append(scored_doc)

        # Sort by combined score
        scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        logger.debug(
            f"Reranked {len(documents)} documents, "
            f"top score: {scored_docs[0]['combined_score']:.3f}"
        )

        return scored_docs

    def _rerank_fallback(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int],
    ) -> List[dict]:
        """Fallback reranking using keyword overlap."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_docs = []
        for doc in documents:
            content = doc.get("content", "").lower()
            content_words = set(content.split())

            # Keyword overlap score
            overlap = len(query_words & content_words)
            total = len(query_words) if query_words else 1
            keyword_score = overlap / total

            # Substring boost
            substring_boost = 0.2 if query_lower in content else 0.0

            fallback_score = keyword_score + substring_boost

            scored_doc = dict(doc)
            scored_doc["reranker_score"] = fallback_score
            original_score = doc.get("score", 0.0)
            scored_doc["combined_score"] = 0.7 * fallback_score + 0.3 * original_score
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        logger.debug(f"Fallback reranked {len(documents)} documents")

        return scored_docs

    def save(self, path: str):
        """Save reranker configuration."""
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"CrossEncoderReranker saved to {path}")

    def load(self, path: str):
        """Load reranker configuration."""
        import json
        from pathlib import Path

        load_path = Path(path)
        config_path = load_path / "config.json"

        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.model_name = config.get("model_name", self.model_name)
            self.max_length = config.get("max_length", self.max_length)
            self.batch_size = config.get("batch_size", self.batch_size)
            self._initialized = False  # Force reload
            logger.info(f"CrossEncoderReranker loaded from {path}")
