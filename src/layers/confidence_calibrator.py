"""Confidence Calibrator — Calibrates confidence scores for better probability estimates.

Implements temperature scaling and Platt scaling to calibrate confidence
scores so they reflect true probabilities (e.g., 0.8 confidence means
the answer is correct 80% of the time).

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
"""

import logging
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("aarag.layers.confidence_calibrator")


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    original_confidence: float
    calibrated_confidence: float
    method: str
    temperature: Optional[float] = None


class TemperatureScaler:
    """Temperature scaling for neural network confidence calibration.

    Learns a single temperature parameter on a validation set to
    calibrate confidence scores.

    Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize temperature scaler.

        Args:
            temperature: Temperature parameter (higher = more uniform)
        """
        self.temperature = temperature
        self._fitted = False

    def fit(
        self,
        confidences: List[float],
        labels: List[int],
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """Fit temperature on validation data.

        Args:
            confidences: Predicted confidence scores
            labels: Ground truth labels (1=correct, 0=incorrect)
            lr: Learning rate for optimization
            max_iter: Maximum iterations
        """
        if len(confidences) != len(labels):
            raise ValueError("confidences and labels must have same length")

        confidences = np.array(confidences, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)

        # Clip confidences to avoid log(0)
        confidences = np.clip(confidences, 1e-7, 1 - 1e-7)

        # Optimize temperature using gradient descent
        temperature = 1.0

        for _ in range(max_iter):
            # Compute calibrated confidences
            scaled = confidences ** (1.0 / temperature)
            calibrated = scaled / (scaled + (1 - confidences) ** (1.0 / temperature))

            # Compute NLL loss
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)
            )

            # Compute gradient
            grad = np.mean(calibrated - labels)

            # Update temperature
            temperature -= lr * grad
            temperature = max(0.1, min(10.0, temperature))

        self.temperature = temperature
        self._fitted = True

        logger.info(f"TemperatureScaler fitted: temperature={temperature:.4f}")

    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score.

        Args:
            confidence: Raw confidence score (0-1)

        Returns:
            Calibrated confidence score (0-1)
        """
        if not self._fitted:
            return confidence

        # Apply temperature scaling
        confidence = max(1e-7, min(1 - 1e-7, confidence))
        scaled = confidence ** (1.0 / self.temperature)
        calibrated = scaled / (scaled + (1 - confidence) ** (1.0 / self.temperature))

        return float(np.clip(calibrated, 0.0, 1.0))

    def save(self, path: str):
        """Save temperature parameter."""
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature, "fitted": self._fitted}, f)

    def load(self, path: str):
        """Load temperature parameter."""
        with open(path, "r") as f:
            data = json.load(f)
        self.temperature = data.get("temperature", 1.0)
        self._fitted = data.get("fitted", False)


class PlattScaler:
    """Platt scaling for confidence calibration.

    Fits a logistic regression model to calibrate confidence scores.
    More flexible than temperature scaling as it has two parameters.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        """Initialize Platt scaler.

        Args:
            a: Scaling parameter
            b: Bias parameter
        """
        self.a = a
        self.b = b
        self._fitted = False

    def fit(
        self,
        confidences: List[float],
        labels: List[int],
        lr: float = 0.01,
        max_iter: int = 200,
    ):
        """Fit Platt scaling parameters on validation data.

        Args:
            confidences: Predicted confidence scores
            labels: Ground truth labels (1=correct, 0=incorrect)
            lr: Learning rate for optimization
            max_iter: Maximum iterations
        """
        if len(confidences) != len(labels):
            raise ValueError("confidences and labels must have same length")

        confidences = np.array(confidences, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)

        # Clip confidences
        confidences = np.clip(confidences, 1e-7, 1 - 1e-7)

        # Convert to logit space
        logits = np.log(confidences / (1 - confidences))

        a, b = 1.0, 0.0

        for _ in range(max_iter):
            # Compute calibrated probabilities
            z = a * logits + b
            z = np.clip(z, -20, 20)  # Prevent overflow
            calibrated = 1.0 / (1.0 + np.exp(-z))

            # Compute gradients
            diff = calibrated - labels
            grad_a = np.mean(diff * logits)
            grad_b = np.mean(diff)

            # Update parameters
            a -= lr * grad_a
            b -= lr * grad_b

            # Regularization
            a = max(0.1, min(10.0, a))
            b = max(-10.0, min(10.0, b))

        self.a = a
        self.b = b
        self._fitted = True

        logger.info(f"PlattScaler fitted: a={a:.4f}, b={b:.4f}")

    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score.

        Args:
            confidence: Raw confidence score (0-1)

        Returns:
            Calibrated confidence score (0-1)
        """
        if not self._fitted:
            return confidence

        confidence = max(1e-7, min(1 - 1e-7, confidence))
        logit = math.log(confidence / (1 - confidence))
        z = self.a * logit + self.b
        z = max(-20, min(20, z))
        calibrated = 1.0 / (1.0 + math.exp(-z))

        return max(0.0, min(1.0, calibrated))

    def save(self, path: str):
        """Save Platt scaling parameters."""
        with open(path, "w") as f:
            json.dump({"a": self.a, "b": self.b, "fitted": self._fitted}, f)

    def load(self, path: str):
        """Load Platt scaling parameters."""
        with open(path, "r") as f:
            data = json.load(f)
        self.a = data.get("a", 1.0)
        self.b = data.get("b", 0.0)
        self._fitted = data.get("fitted", False)


class ConfidenceCalibrator:
    """Multi-layer confidence calibrator.

    Calibrates confidence scores from different pipeline layers
    using per-layer calibration models.

    Example:
        >>> calibrator = ConfidenceCalibrator()
        >>> calibrator.fit_layer("router", confidences, labels)
        >>> calibrated = calibrator.calibrate("router", 0.85)
        """

    def __init__(self, method: str = "temperature"):
        """Initialize confidence calibrator.

        Args:
            method: Calibration method ("temperature" or "platt")
        """
        self.method = method
        self._scalers: Dict[str, any] = {}
        self._fitted_layers: set = set()

        logger.info(f"ConfidenceCalibrator initialized (method={method})")

    def fit_layer(
        self,
        layer_name: str,
        confidences: List[float],
        labels: List[int],
        method: Optional[str] = None,
    ):
        """Fit calibration for a specific layer.

        Args:
            layer_name: Name of the layer (e.g., "router", "crag", "reflection")
            confidences: Predicted confidence scores from validation
            labels: Ground truth labels (1=correct, 0=incorrect)
            method: Override calibration method for this layer
        """
        use_method = method or self.method

        if use_method == "temperature":
            scaler = TemperatureScaler()
        elif use_method == "platt":
            scaler = PlattScaler()
        else:
            raise ValueError(f"Unknown calibration method: {use_method}")

        scaler.fit(confidences, labels)
        self._scalers[layer_name] = scaler
        self._fitted_layers.add(layer_name)

        logger.info(f"Fitted {use_method} calibrator for layer '{layer_name}'")

    def calibrate(self, layer_name: str, confidence: float) -> float:
        """Calibrate a confidence score for a specific layer.

        Args:
            layer_name: Name of the layer
            confidence: Raw confidence score

        Returns:
            Calibrated confidence score (0-1)
        """
        if layer_name not in self._scalers:
            logger.debug(f"No calibrator for layer '{layer_name}', returning raw")
            return confidence

        calibrated = self._scalers[layer_name].calibrate(confidence)

        logger.debug(
            f"Calibrated {layer_name}: {confidence:.4f} → {calibrated:.4f}"
        )

        return calibrated

    def calibrate_all(self, layer_confidences: Dict[str, float]) -> Dict[str, float]:
        """Calibrate confidences for all layers.

        Args:
            layer_confidences: Dict of layer_name -> confidence

        Returns:
            Dict of layer_name -> calibrated_confidence
        """
        calibrated = {}
        for layer_name, confidence in layer_confidences.items():
            calibrated[layer_name] = self.calibrate(layer_name, confidence)
        return calibrated

    def compute_overall_confidence(
        self,
        layer_confidences: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted overall confidence from all layers.

        Args:
            layer_confidences: Dict of layer_name -> confidence
            weights: Optional weights per layer (default: equal weights)

        Returns:
            Weighted average of calibrated confidences
        """
        if not layer_confidences:
            return 0.0

        # Calibrate all confidences
        calibrated = self.calibrate_all(layer_confidences)

        # Compute weighted average
        if weights is None:
            weights = {k: 1.0 for k in calibrated.keys()}

        total_weight = 0.0
        weighted_sum = 0.0

        for layer_name, conf in calibrated.items():
            w = weights.get(layer_name, 1.0)
            weighted_sum += w * conf
            total_weight += w

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    @property
    def is_fitted(self) -> bool:
        """Check if any layers have been fitted."""
        return len(self._fitted_layers) > 0

    @property
    def fitted_layers(self) -> List[str]:
        """List of fitted layer names."""
        return list(self._fitted_layers)

    def save(self, path: str):
        """Save all calibrators."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "method": self.method,
            "fitted_layers": list(self._fitted_layers),
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save individual scalers
        for layer_name, scaler in self._scalers.items():
            scaler.save(str(save_path / f"{layer_name}_scaler.json"))

        logger.info(f"ConfidenceCalibrator saved to {path}")

    def load(self, path: str):
        """Load all calibrators."""
        load_path = Path(path)

        # Load config
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.method = config.get("method", self.method)
            self._fitted_layers = set(config.get("fitted_layers", []))

        # Load individual scalers
        for layer_name in self._fitted_layers:
            scaler_path = load_path / f"{layer_name}_scaler.json"
            if scaler_path.exists():
                if self.method == "temperature":
                    scaler = TemperatureScaler()
                else:
                    scaler = PlattScaler()
                scaler.load(str(scaler_path))
                self._scalers[layer_name] = scaler

        logger.info(f"ConfidenceCalibrator loaded from {path}")
