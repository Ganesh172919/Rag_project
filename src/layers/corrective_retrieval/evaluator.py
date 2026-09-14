"""Retrieval Evaluator — Assess quality of retrieved documents."""

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class RetrievalEvaluator:
    """Evaluate whether retrieved documents are relevant to the query.

    Based on: CRAG (Yan et al., ICML 2024)
    Uses a lightweight classifier to score retrieval quality.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-small", device: str = "auto"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model = None
        self._tokenizer = None
        self._use_transformer = False

    @property
    def model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,  # relevant / irrelevant
                ).to(self.device)
                self._model.eval()
                self._use_transformer = True
            except Exception as e:
                print(f"Warning: Could not load evaluator model: {e}")
                self._use_transformer = False
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                self._use_transformer = False
        return self._tokenizer

    def score(self, query: str, documents: List[str]) -> float:
        """Score the relevance of retrieved documents to the query.

        Args:
            query: The user's question
            documents: List of retrieved document texts

        Returns:
            Confidence score between 0 and 1.
            > 0.8: CORRECT (use as-is)
            0.3-0.8: AMBIGUOUS (combine + filter)
            < 0.3: INCORRECT (fallback to web search)
        """
        if not documents:
            return 0.0

        if self._use_transformer and self._model is not None:
            return self._score_transformer(query, documents)
        else:
            return self._score_heuristic(query, documents)

    def score_per_document(self, query: str, documents: List[str]) -> List[float]:
        """Score each document individually."""
        if not documents:
            return []

        if self._use_transformer and self._model is not None:
            return [self._score_single_transformer(query, doc) for doc in documents]
        else:
            return [self._score_single_heuristic(query, doc) for doc in documents]

    def _score_transformer(self, query: str, documents: List[str]) -> float:
        """Score using fine-tuned transformer."""
        import torch

        # Score each document and take the max
        scores = []
        for doc in documents:
            score = self._score_single_transformer(query, doc)
            scores.append(score)

        # Weighted combination: max score matters most, but average also counts
        if scores:
            return 0.7 * max(scores) + 0.3 * np.mean(scores)
        return 0.0

    def _score_single_transformer(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        import torch

        inputs = self.tokenizer(
            query, document,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return float(probs[1])  # Probability of "relevant" class

    def _score_heuristic(self, query: str, documents: List[str]) -> float:
        """Score using keyword overlap heuristic (fallback)."""
        import string
        if not documents:
            return 0.0

        query_words = set(query.lower().translate(str.maketrans("", "", string.punctuation)).split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                      "when", "where", "how", "which", "that", "this", "it", "of", "in", "to", "and", "or", "be"}
        query_words -= stop_words

        if not query_words:
            return 0.5

        scores = []
        for doc in documents:
            doc_lower = doc.lower().translate(str.maketrans("", "", string.punctuation))
            doc_words = set(doc_lower.split())
            overlap = len(query_words & doc_words)
            substring_boost = sum(0.1 for w in query_words if w in doc_lower)
            score = (overlap / len(query_words)) + substring_boost
            scores.append(min(score, 1.0))

        return 0.7 * max(scores) + 0.3 * np.mean(scores) if scores else 0.0

    def _score_single_heuristic(self, query: str, document: str) -> float:
        """Score a single document using heuristics."""
        import string
        query_words = set(query.lower().translate(str.maketrans("", "", string.punctuation)).split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                      "when", "where", "how", "which", "that", "this", "it", "of", "in", "to", "and", "or", "be"}
        query_words -= stop_words

        if not query_words:
            return 0.5

        doc_lower = document.lower().translate(str.maketrans("", "", string.punctuation))
        doc_words = set(doc_lower.split())
        overlap = len(query_words & doc_words)
        substring_boost = sum(0.1 for w in query_words if w in doc_lower)
        return min((overlap / len(query_words)) + substring_boost, 1.0)

    def save(self, path: str):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        if self._use_transformer and self._model is not None:
            self._model.save_pretrained(str(save_path))
            self._tokenizer.save_pretrained(str(save_path))

    def load(self, path: str):
        load_path = Path(path)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._model = AutoModelForSequenceClassification.from_pretrained(str(load_path)).to(self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            self._model.eval()
            self._use_transformer = True
        except Exception as e:
            print(f"Warning: Could not load evaluator: {e}")
            self._use_transformer = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
