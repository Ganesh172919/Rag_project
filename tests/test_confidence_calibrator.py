"""Tests for ConfidenceCalibrator."""

import pytest
from src.layers.confidence_calibrator import (
    TemperatureScaler,
    PlattScaler,
    ConfidenceCalibrator,
)


class TestTemperatureScaler:
    """Tests for temperature scaling."""

    @pytest.fixture
    def scaler(self):
        """Create a temperature scaler."""
        return TemperatureScaler()

    def test_default_temperature(self, scaler):
        """Default temperature should be 1.0."""
        assert scaler.temperature == 1.0

    def test_identity_without_fitting(self, scaler):
        """Without fitting, should return input unchanged."""
        assert scaler.calibrate(0.8) == 0.8
        assert scaler.calibrate(0.5) == 0.5

    def test_fit_with_data(self, scaler):
        """Should be able to fit with validation data."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        labels = [1, 1, 1, 0, 0, 0, 0]
        scaler.fit(confidences, labels)
        assert scaler._fitted is True
        assert scaler.temperature > 0

    def test_calibrate_after_fit(self, scaler):
        """Calibration should adjust confidences after fitting."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        labels = [1, 1, 1, 0, 0, 0, 0]
        scaler.fit(confidences, labels)

        # High confidence should remain high
        calibrated = scaler.calibrate(0.9)
        assert 0.5 < calibrated <= 1.0

    def test_save_load(self, scaler, tmp_path):
        """Should be able to save and load."""
        scaler.temperature = 1.5
        scaler._fitted = True
        scaler.save(str(tmp_path / "scaler.json"))

        new_scaler = TemperatureScaler()
        new_scaler.load(str(tmp_path / "scaler.json"))
        assert new_scaler.temperature == 1.5
        assert new_scaler._fitted is True


class TestPlattScaler:
    """Tests for Platt scaling."""

    @pytest.fixture
    def scaler(self):
        """Create a Platt scaler."""
        return PlattScaler()

    def test_default_params(self, scaler):
        """Default parameters should be a=1, b=0."""
        assert scaler.a == 1.0
        assert scaler.b == 0.0

    def test_identity_without_fitting(self, scaler):
        """Without fitting, should return input unchanged."""
        assert scaler.calibrate(0.8) == 0.8

    def test_fit_with_data(self, scaler):
        """Should be able to fit with validation data."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        labels = [1, 1, 1, 0, 0, 0, 0]
        scaler.fit(confidences, labels)
        assert scaler._fitted is True

    def test_calibrate_after_fit(self, scaler):
        """Calibration should adjust confidences after fitting."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        labels = [1, 1, 1, 0, 0, 0, 0]
        scaler.fit(confidences, labels)

        calibrated = scaler.calibrate(0.9)
        assert 0.0 <= calibrated <= 1.0

    def test_save_load(self, scaler, tmp_path):
        """Should be able to save and load."""
        scaler.a = 2.0
        scaler.b = -0.5
        scaler._fitted = True
        scaler.save(str(tmp_path / "scaler.json"))

        new_scaler = PlattScaler()
        new_scaler.load(str(tmp_path / "scaler.json"))
        assert new_scaler.a == 2.0
        assert new_scaler.b == -0.5


class TestConfidenceCalibrator:
    """Tests for the multi-layer calibrator."""

    @pytest.fixture
    def calibrator(self):
        """Create a confidence calibrator."""
        return ConfidenceCalibrator(method="temperature")

    def test_init(self, calibrator):
        """Should initialize with specified method."""
        assert calibrator.method == "temperature"
        assert not calibrator.is_fitted

    def test_fit_layer(self, calibrator):
        """Should be able to fit individual layers."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 1, 0, 0, 0]
        calibrator.fit_layer("router", confidences, labels)
        assert "router" in calibrator.fitted_layers

    def test_calibrate_layer(self, calibrator):
        """Should calibrate using fitted layer."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 1, 0, 0, 0]
        calibrator.fit_layer("router", confidences, labels)

        calibrated = calibrator.calibrate("router", 0.85)
        assert 0.0 <= calibrated <= 1.0

    def test_calibrate_unfitted_layer(self, calibrator):
        """Should return raw confidence for unfitted layers."""
        assert calibrator.calibrate("unknown", 0.85) == 0.85

    def test_calibrate_all(self, calibrator):
        """Should calibrate multiple layers at once."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 1, 0, 0, 0]
        calibrator.fit_layer("router", confidences, labels)
        calibrator.fit_layer("crag", confidences, labels)

        result = calibrator.calibrate_all({
            "router": 0.85,
            "crag": 0.75,
            "unknown": 0.65,
        })

        assert "router" in result
        assert "crag" in result
        assert result["unknown"] == 0.65  # Unfitted, returns raw

    def test_compute_overall_confidence(self, calibrator):
        """Should compute weighted overall confidence."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 1, 0, 0, 0]
        calibrator.fit_layer("router", confidences, labels)

        overall = calibrator.compute_overall_confidence({
            "router": 0.85,
            "crag": 0.75,
        })

        assert 0.0 <= overall <= 1.0

    def test_save_load(self, calibrator, tmp_path):
        """Should be able to save and load."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 1, 0, 0, 0]
        calibrator.fit_layer("router", confidences, labels)

        calibrator.save(str(tmp_path / "calibrator"))

        new_calibrator = ConfidenceCalibrator()
        new_calibrator.load(str(tmp_path / "calibrator"))
        assert "router" in new_calibrator.fitted_layers
