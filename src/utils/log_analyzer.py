"""Log Analyzer — Parses and analyzes AARAG structured logs.

Provides:
- Query performance profiling
- Layer bottleneck detection
- Error pattern identification
- Summary report generation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("aarag.utils.log_analyzer")


@dataclass
class QueryProfile:
    """Performance profile for a single query."""
    query_id: str
    query: str
    total_ms: float
    layer_times: Dict[str, float]
    strategy: str
    confidence: float
    status: str
    errors: List[str]


@dataclass
class AnalysisReport:
    """Log analysis report."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    slowest_queries: List[QueryProfile]
    layer_bottlenecks: Dict[str, float]
    error_patterns: List[Tuple[str, int]]
    strategy_distribution: Dict[str, int]
    hourly_distribution: Dict[str, int]


class LogAnalyzer:
    """Analyzes AARAG structured logs.

    Parses JSONL log files and generates performance reports
    including bottleneck detection and error analysis.

    Example:
        >>> analyzer = LogAnalyzer()
        >>> report = analyzer.analyze_file("logs/aarag.jsonl")
        >>> analyzer.print_report(report)
        >>> analyzer.save_report(report, "results/log_analysis.json")
    """

    def __init__(self):
        """Initialize log analyzer."""
        logger.info("LogAnalyzer initialized")

    def parse_log_file(self, file_path: str) -> List[Dict]:
        """Parse a JSONL log file.

        Args:
            file_path: Path to JSONL log file

        Returns:
            List of parsed log entries
        """
        entries = []
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"Log file not found: {file_path}")
            return entries

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    logger.debug(f"Skipping invalid JSON at line {line_num}")

        logger.info(f"Parsed {len(entries)} entries from {file_path}")
        return entries

    def analyze_entries(self, entries: List[Dict]) -> AnalysisReport:
        """Analyze parsed log entries.

        Args:
            entries: List of parsed log entries

        Returns:
            AnalysisReport with aggregated statistics
        """
        # Group by query_id
        queries: Dict[str, Dict] = defaultdict(lambda: {
            "query": "",
            "total_ms": 0.0,
            "layer_times": {},
            "strategy": "",
            "confidence": 0.0,
            "status": "unknown",
            "errors": [],
            "timestamp": "",
        })

        for entry in entries:
            event = entry.get("event", "")
            query_id = entry.get("query_id", "unknown")

            if event == "query_start":
                queries[query_id]["query"] = entry.get("query", "")
                queries[query_id]["timestamp"] = entry.get("timestamp", "")

            elif event == "layer_start":
                layer = entry.get("layer_name", "")
                queries[query_id]["layer_times"][layer] = time.time()

            elif event == "layer_end":
                layer = entry.get("layer_name", "")
                elapsed = entry.get("elapsed_ms", 0.0)
                queries[query_id]["layer_times"][layer] = elapsed

            elif event == "query_end":
                queries[query_id]["total_ms"] = entry.get("total_ms", 0.0)
                queries[query_id]["strategy"] = entry.get("strategy", "")
                queries[query_id]["confidence"] = entry.get("confidence", 0.0)
                queries[query_id]["status"] = "success"

            elif event == "error":
                queries[query_id]["errors"].append(entry.get("error_message", ""))
                queries[query_id]["status"] = "error"

        # Build profiles
        profiles = []
        for query_id, data in queries.items():
            profiles.append(QueryProfile(
                query_id=query_id,
                query=data["query"],
                total_ms=data["total_ms"],
                layer_times=data["layer_times"],
                strategy=data["strategy"],
                confidence=data["confidence"],
                status=data["status"],
                errors=data["errors"],
            ))

        # Compute statistics
        successful = [p for p in profiles if p.status == "success"]
        failed = [p for p in profiles if p.status != "success"]

        latencies = [p.total_ms for p in successful]

        # Layer bottleneck analysis
        layer_times = defaultdict(list)
        for p in successful:
            for layer, elapsed in p.layer_times.items():
                if isinstance(elapsed, (int, float)) and elapsed > 0:
                    layer_times[layer].append(elapsed)

        layer_bottlenecks = {}
        for layer, times in layer_times.items():
            layer_bottlenecks[layer] = sum(times) / len(times) if times else 0.0

        # Error patterns
        error_counter = Counter()
        for p in failed:
            for error in p.errors:
                # Extract error type
                error_type = error.split(":")[0] if ":" in error else error[:50]
                error_counter[error_type] += 1

        # Strategy distribution
        strategy_counter = Counter(p.strategy for p in successful if p.strategy)

        # Hourly distribution
        hourly_counter = Counter()
        for p in profiles:
            if p.query_id:
                # Extract hour from timestamp if available
                for entry in entries:
                    if entry.get("query_id") == p.query_id and entry.get("timestamp"):
                        hour = entry["timestamp"][:13]  # YYYY-MM-DDTHH
                        hourly_counter[hour] += 1
                        break

        # Sort latencies for percentiles
        latencies_sorted = sorted(latencies) if latencies else [0]

        import numpy as np

        return AnalysisReport(
            total_queries=len(profiles),
            successful_queries=len(successful),
            failed_queries=len(failed),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency_ms=float(np.percentile(latencies_sorted, 50)) if latencies else 0,
            p95_latency_ms=float(np.percentile(latencies_sorted, 95)) if latencies else 0,
            p99_latency_ms=float(np.percentile(latencies_sorted, 99)) if latencies else 0,
            slowest_queries=sorted(profiles, key=lambda p: p.total_ms, reverse=True)[:10],
            layer_bottlenecks=layer_bottlenecks,
            error_patterns=error_counter.most_common(10),
            strategy_distribution=dict(strategy_counter),
            hourly_distribution=dict(hourly_counter),
        )

    def analyze_file(self, file_path: str) -> AnalysisReport:
        """Analyze a log file.

        Args:
            file_path: Path to log file

        Returns:
            AnalysisReport
        """
        entries = self.parse_log_file(file_path)
        return self.analyze_entries(entries)

    def analyze_directory(self, dir_path: str) -> AnalysisReport:
        """Analyze all log files in a directory.

        Args:
            dir_path: Path to directory containing log files

        Returns:
            AnalysisReport
        """
        all_entries = []
        path = Path(dir_path)

        for log_file in path.glob("*.jsonl"):
            all_entries.extend(self.parse_log_file(str(log_file)))

        for log_file in path.glob("*.log"):
            all_entries.extend(self.parse_log_file(str(log_file)))

        return self.analyze_entries(all_entries)

    def save_report(self, report: AnalysisReport, output_path: str):
        """Save analysis report to JSON.

        Args:
            report: AnalysisReport
            output_path: Path to save report
        """
        data = {
            "summary": {
                "total_queries": report.total_queries,
                "successful": report.successful_queries,
                "failed": report.failed_queries,
                "avg_latency_ms": round(report.avg_latency_ms, 2),
                "p50_latency_ms": round(report.p50_latency_ms, 2),
                "p95_latency_ms": round(report.p95_latency_ms, 2),
                "p99_latency_ms": round(report.p99_latency_ms, 2),
            },
            "layer_bottlenecks": {
                k: round(v, 2) for k, v in report.layer_bottlenecks.items()
            },
            "error_patterns": [
                {"pattern": p, "count": c} for p, c in report.error_patterns
            ],
            "strategy_distribution": report.strategy_distribution,
            "slowest_queries": [
                {
                    "query_id": q.query_id,
                    "query": q.query[:100],
                    "total_ms": round(q.total_ms, 2),
                    "strategy": q.strategy,
                    "confidence": round(q.confidence, 4),
                    "status": q.status,
                }
                for q in report.slowest_queries
            ],
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis report saved to {output_path}")

    def print_report(self, report: AnalysisReport):
        """Print analysis report to console."""
        print("\n" + "=" * 60)
        print("  LOG ANALYSIS REPORT")
        print("=" * 60)

        print(f"\n  Queries: {report.total_queries} total")
        print(f"    Successful: {report.successful_queries}")
        print(f"    Failed: {report.failed_queries}")

        print(f"\n  Latency:")
        print(f"    Mean:  {report.avg_latency_ms:.2f}ms")
        print(f"    p50:   {report.p50_latency_ms:.2f}ms")
        print(f"    p95:   {report.p95_latency_ms:.2f}ms")
        print(f"    p99:   {report.p99_latency_ms:.2f}ms")

        print(f"\n  Layer Bottlenecks (avg ms):")
        for layer, avg_ms in sorted(
            report.layer_bottlenecks.items(), key=lambda x: -x[1]
        ):
            print(f"    {layer:<25} {avg_ms:>8.2f}ms")

        print(f"\n  Strategy Distribution:")
        for strategy, count in report.strategy_distribution.items():
            print(f"    {strategy:<20} {count:>4}")

        if report.error_patterns:
            print(f"\n  Error Patterns:")
            for pattern, count in report.error_patterns:
                print(f"    {pattern:<40} {count:>4}")

        if report.slowest_queries:
            print(f"\n  Slowest Queries:")
            for q in report.slowest_queries[:5]:
                print(f"    {q.total_ms:>8.2f}ms | {q.strategy:<15} | {q.query[:50]}...")

        print("=" * 60)


# Import time for timestamps
import time
