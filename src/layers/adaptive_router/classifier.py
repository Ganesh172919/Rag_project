"""Query Complexity Classifier — Classify queries into complexity levels."""

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

from .feature_extractor import FeatureExtractor


class QueryComplexityClassifier:
    """Classify query complexity into 4 levels:
    0: No-retrieval (simple factual, LLM can answer directly)
    1: Single-step RAG (needs one retrieval hop)
    2: Multi-step RAG (needs multiple retrieval hops)
    3: Graph-global (needs corpus-level reasoning)
    """

    LABEL_NAMES = ["no_retrieval", "single_step", "multi_step", "graph_global"]

    def __init__(self, model_name: str = "microsoft/deberta-v3-base", device: str = "auto"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.feature_extractor = FeatureExtractor()
        self._model = None
        self._tokenizer = None
        self._use_transformer = False
        self._fallback_classifier = _RuleBasedClassifier()

    @property
    def model(self):
        """Lazy-load the transformer model."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=4,
                    problem_type="single_label_classification",
                ).to(self.device)
                self._model.eval()
                self._use_transformer = True
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to rule-based classifier.")
                self._use_transformer = False
        return self._model

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                self._use_transformer = False
        return self._tokenizer

    def predict(self, query: str) -> Tuple[int, float]:
        """Predict complexity level for a single query.

        Returns:
            (level, confidence) where level is 0-3 and confidence is 0-1
        """
        if self._use_transformer and self._model is not None:
            return self._predict_transformer(query)
        else:
            return self._predict_rule_based(query)

    def predict_batch(self, queries: List[str]) -> List[Tuple[int, float]]:
        """Predict complexity levels for a batch of queries."""
        return [self.predict(q) for q in queries]

    def _predict_transformer(self, query: str) -> Tuple[int, float]:
        """Predict using fine-tuned transformer."""
        import torch

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        level = int(np.argmax(probs))
        confidence = float(probs[level])
        return level, confidence

    def _predict_rule_based(self, query: str) -> Tuple[int, float]:
        """Predict using rule-based heuristics (fallback)."""
        return self._fallback_classifier.predict(query)

    def save(self, path: str):
        """Save the classifier model."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._use_transformer and self._model is not None:
            self._model.save_pretrained(str(save_path))
            self._tokenizer.save_pretrained(str(save_path))

        # Save config
        config = {
            "model_name": self.model_name,
            "use_transformer": self._use_transformer,
        }
        with open(save_path / "classifier_config.json", "w") as f:
            json.dump(config, f)

    def load(self, path: str):
        """Load the classifier model."""
        load_path = Path(path)
        config_path = load_path / "classifier_config.json"

        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self._use_transformer = config.get("use_transformer", False)

        if self._use_transformer:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._model = AutoModelForSequenceClassification.from_pretrained(str(load_path)).to(self.device)
                self._tokenizer = AutoTokenizer.from_pretrained(str(load_path))
                self._model.eval()
            except Exception as e:
                print(f"Warning: Could not load transformer: {e}")
                self._use_transformer = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device


class _RuleBasedClassifier:
    """Rule-based fallback classifier using feature heuristics."""

    def predict(self, query: str) -> Tuple[int, float]:
        """Predict complexity using rules."""
        extractor = FeatureExtractor()
        features = extractor.extract(query)
        query_lower = query.lower().strip()
        words = query_lower.split()

        # Level 0: Simple factual questions
        simple_triggers = ["who is", "who was", "what is", "what was", "when did",
                          "when was", "where is", "where was", "how old", "how many"]
        if any(query_lower.startswith(t) for t in simple_triggers) and features["word_count"] <= 12:
            return 0, 0.85

        # Level 3: Global/corpus-level questions
        if features["global_keyword_count"] >= 2 or features["has_summary"]:
            return 3, 0.80

        # Level 2: Multi-hop questions
        if features["multi_hop_keyword_count"] >= 2 or features["has_comparison"]:
            return 2, 0.75
        if features["conjunction_count"] >= 2 and features["word_count"] > 15:
            return 2, 0.70

        # Level 1: Standard retrieval questions (default)
        if features["word_count"] > 8:
            return 1, 0.65

        # Short questions that aren't simple
        return 1, 0.55
