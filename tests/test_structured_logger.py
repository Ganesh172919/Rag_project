"""Tests for StructuredLogger."""

import json
import pytest
from pathlib import Path
from src.utils.structured_logger import StructuredLogger, LogContext


class TestStructuredLogger:
    """Tests for the structured logger."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create a structured logger with file output."""
        return StructuredLogger(
            name="test",
            log_dir=str(tmp_path / "logs"),
            level="DEBUG",
            json_output=True,
            console_output=False,
        )

    def test_init(self, logger):
        """Should initialize correctly."""
        assert logger.name == "test"
        assert logger.json_output is True

    def test_log_info(self, logger, tmp_path):
        """Should log info messages."""
        logger.info("Test message", key="value")

        log_file = tmp_path / "logs" / "test.jsonl"
        assert log_file.exists()

        with open(log_file, "r") as f:
            entry = json.loads(f.readline())
            assert entry["level"] == "INFO"
            assert entry["message"] == "Test message"
            assert entry["key"] == "value"

    def test_log_levels(self, logger):
        """Should support all log levels."""
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")
        logger.pipeline("pipeline message")

    def test_context_push_pop(self, logger):
        """Should support context push/pop."""
        logger.push_context(query_id="q123", layer="router")
        logger.info("In context")

        logger.pop_context()
        logger.info("Out of context")

    def test_performance_counter(self, logger):
        """Should track performance counters."""
        import time

        logger.start_counter("test_counter")
        time.sleep(0.01)
        elapsed = logger.stop_counter("test_counter")

        assert elapsed > 0

    def test_log_layer_start_end(self, logger):
        """Should log layer start and end."""
        logger.log_layer_start(2, "Adaptive Router", query_id="q123")
        logger.log_layer_end(2, "Adaptive Router", query_id="q123")

    def test_log_query_start_end(self, logger):
        """Should log query start and end."""
        query_id = logger.log_query_start("What is ML?")
        assert query_id is not None

        logger.log_query_end(
            query_id=query_id,
            answer="Machine learning is...",
            confidence=0.85,
            strategy="multi_step",
        )

    def test_log_decision(self, logger):
        """Should log decisions."""
        logger.log_decision("strategy", "multi_step", 0.85)

    def test_log_error(self, logger):
        """Should log errors."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            logger.log_error(e, context="test")

    def test_reset_counters(self, logger):
        """Should reset counters."""
        logger.start_counter("test")
        logger.reset_counters()
        assert logger.get_counter("test") is None
