"""Tests for LogAnalyzer."""

import json
import pytest
from pathlib import Path
from src.utils.log_analyzer import LogAnalyzer, QueryProfile, AnalysisReport


class TestLogAnalyzer:
    """Tests for the log analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a log analyzer."""
        return LogAnalyzer()

    @pytest.fixture
    def sample_log_entries(self):
        """Sample log entries for testing."""
        return [
            {
                "timestamp": "2026-07-12T10:00:00Z",
                "level": "INFO",
                "event": "query_start",
                "query_id": "q001",
                "query": "What is machine learning?",
            },
            {
                "timestamp": "2026-07-12T10:00:00.5Z",
                "level": "INFO",
                "event": "layer_start",
                "query_id": "q001",
                "layer_name": "router",
            },
            {
                "timestamp": "2026-07-12T10:00:00.6Z",
                "level": "INFO",
                "event": "layer_end",
                "query_id": "q001",
                "layer_name": "router",
                "elapsed_ms": 100,
            },
            {
                "timestamp": "2026-07-12T10:00:02Z",
                "level": "INFO",
                "event": "query_end",
                "query_id": "q001",
                "total_ms": 2000,
                "strategy": "multi_step",
                "confidence": 0.85,
            },
        ]

    @pytest.fixture
    def log_file(self, tmp_path, sample_log_entries):
        """Create a sample log file."""
        log_path = tmp_path / "test.jsonl"
        with open(log_path, "w") as f:
            for entry in sample_log_entries:
                f.write(json.dumps(entry) + "\n")
        return str(log_path)

    def test_parse_log_file(self, analyzer, log_file):
        """Should parse JSONL log files."""
        entries = analyzer.parse_log_file(log_file)
        assert len(entries) == 4

    def test_parse_nonexistent_file(self, analyzer):
        """Should handle nonexistent files gracefully."""
        entries = analyzer.parse_log_file("nonexistent.jsonl")
        assert entries == []

    def test_analyze_entries(self, analyzer, sample_log_entries):
        """Should analyze log entries correctly."""
        report = analyzer.analyze_entries(sample_log_entries)
        assert report.total_queries == 1
        assert report.successful_queries == 1
        assert report.failed_queries == 0

    def test_analyze_file(self, analyzer, log_file):
        """Should analyze a log file."""
        report = analyzer.analyze_file(log_file)
        assert report.total_queries == 1
        assert report.avg_latency_ms > 0

    def test_report_has_bottlenecks(self, analyzer, sample_log_entries):
        """Report should include layer bottlenecks."""
        report = analyzer.analyze_entries(sample_log_entries)
        assert isinstance(report.layer_bottlenecks, dict)

    def test_report_has_strategy_distribution(self, analyzer, sample_log_entries):
        """Report should include strategy distribution."""
        report = analyzer.analyze_entries(sample_log_entries)
        assert "multi_step" in report.strategy_distribution

    def test_save_report(self, analyzer, sample_log_entries, tmp_path):
        """Should save report to JSON."""
        report = analyzer.analyze_entries(sample_log_entries)
        output_path = str(tmp_path / "analysis.json")
        analyzer.save_report(report, output_path)

        assert Path(output_path).exists()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert "summary" in data
        assert "layer_bottlenecks" in data

    def test_empty_entries(self, analyzer):
        """Should handle empty entries gracefully."""
        report = analyzer.analyze_entries([])
        assert report.total_queries == 0
        assert report.avg_latency_ms == 0

    def test_error_entries(self, analyzer):
        """Should handle error entries."""
        entries = [
            {
                "event": "query_start",
                "query_id": "q001",
                "query": "test",
            },
            {
                "event": "error",
                "query_id": "q001",
                "error_message": "Something went wrong",
            },
        ]
        report = analyzer.analyze_entries(entries)
        assert report.failed_queries == 1
