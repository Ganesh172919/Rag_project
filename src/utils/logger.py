"""Logger — Structured logging for AARAG with Rich formatting.

Supports two output modes:
- Rich console output (default) — colorful, formatted terminal output
- JSON structured output — machine-parseable JSON lines for log aggregation

Also provides performance timing decorators and structured context fields.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime

from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


# ──────────────────────────────────────────────
# Custom log levels for pipeline stages
# ──────────────────────────────────────────────
PIPELINE = 25  # Between INFO and WARNING
logging.addLevelName(PIPELINE, "PIPELINE")


def pipeline(self, message, *args, **kwargs):
    if self.isEnabledFor(PIPELINE):
        self._log(PIPELINE, message, args, **kwargs)


logging.Logger.pipeline = pipeline


# ──────────────────────────────────────────────
# JSON formatter for structured logging
# ──────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        for key in ["query_id", "layer", "component", "latency_ms", "confidence"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ──────────────────────────────────────────────
# Logger factory
# ──────────────────────────────────────────────
def get_logger(
    name: str = "aarag",
    level: str = "DEBUG",
    log_file: Optional[str] = None,
    json_file: Optional[str] = None,
) -> logging.Logger:
    """Get a configured logger with Rich console output and optional file logging.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, PIPELINE, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for text file logging
        json_file: Optional file path for JSON structured logging

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    # Rich console handler — colorful, formatted output
    console_handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        show_level=True,
        markup=True,
    )
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # File handler — detailed text logs for debugging
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8", mode="a")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # JSON file handler — structured logs for log aggregation
    if json_file:
        json_path = Path(json_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(str(json_path), encoding="utf-8", mode="a")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    return logger


def log_section(logger, title: str, level: int = logging.INFO):
    """Log a formatted section header."""
    logger.log(level, f"{'─' * 60}")
    logger.log(level, f"  {title}")
    logger.log(level, f"{'─' * 60}")


def log_table(logger, title: str, headers: list, rows: list, level: int = logging.INFO):
    """Log a formatted table."""
    table = Table(title=title, box=box.ROUNDED)
    for h in headers:
        table.add_column(h, style="cyan")
    for row in rows:
        table.add_row(*[str(v) for v in row])
    console = Console()
    console.print(table)


def log_layer_start(logger, layer_num: int, layer_name: str):
    """Log the start of a pipeline layer."""
    logger.log(PIPELINE, f"[bold blue]▸ LAYER {layer_num}: {layer_name}[/bold blue]", extra={"markup": True})


def log_layer_end(logger, layer_num: int, layer_name: str, elapsed: float):
    """Log the end of a pipeline layer."""
    logger.log(PIPELINE, f"[bold green]✓ LAYER {layer_num}: {layer_name} completed in {elapsed:.3f}s[/bold green]", extra={"markup": True})


def log_decision(logger, decision_type: str, value: str, confidence: float = None):
    """Log a routing/decision point."""
    conf_str = f" (confidence: {confidence:.2f})" if confidence is not None else ""
    logger.info(f"  → {decision_type}: {value}{conf_str}")


def log_metric(logger, metric_name: str, value: float, threshold: float = None):
    """Log a metric value with optional threshold comparison."""
    if threshold is not None:
        status = "✓" if value >= threshold else "✗"
        logger.info(f"  → {metric_name}: {value:.4f} (threshold: {threshold}) {status}")
    else:
        logger.info(f"  → {metric_name}: {value:.4f}")


# ──────────────────────────────────────────────
# Performance timing decorators
# ──────────────────────────────────────────────
def timer(func=None, *, logger_name: str = "aarag.timer"):
    """Decorator to time function execution.

    Can be used as:
        @timer
        def my_function():
            pass

        @timer(logger_name="aarag.router")
        def my_function():
            pass
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - start

            fn_logger = logging.getLogger(logger_name)
            fn_logger.debug(
                f"{fn.__name__} completed in {elapsed:.4f}s"
            )

            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class PerformanceTimer:
    """Context manager for timing code blocks.

    Example:
        >>> with PerformanceTimer("retrieval") as t:
        ...     # code to time
        ...     pass
        >>> print(f"Elapsed: {t.elapsed:.4f}s")
    """

    def __init__(self, name: str, logger_name: str = "aarag.timer"):
        self.name = name
        self.logger_name = logger_name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        fn_logger = logging.getLogger(self.logger_name)
        fn_logger.debug(f"{self.name} completed in {self.elapsed:.4f}s")
        return False


# Default logger
logger = get_logger()
