"""Structured Logger — JSON structured logging for AARAG.

Provides machine-parseable JSON logs with:
- Context propagation (query_id, session_id, layer)
- Performance counters (layer timings, token counts)
- Structured fields for log aggregation
- Compatible with ELK, Datadog, CloudWatch, etc.
"""

import json
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class LogContext:
    """Context for structured logging."""
    query_id: Optional[str] = None
    session_id: Optional[str] = None
    layer: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class PerformanceCounter:
    """Performance counter for tracking metrics."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    value: float = 0.0
    unit: str = "ms"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return self.value


class StructuredLogger:
    """Structured JSON logger for AARAG.

    Outputs logs in JSON format for easy parsing by log aggregation tools.

    Example:
        >>> logger = StructuredLogger("aarag", log_dir="logs/")
        >>> logger.info("Query processed", extra={
        ...     "query_id": "q123",
        ...     "latency_ms": 150,
        ...     "strategy": "multi_step",
        ... })
    """

    def __init__(
        self,
        name: str = "aarag",
        log_dir: Optional[str] = None,
        level: str = "INFO",
        json_output: bool = True,
        console_output: bool = True,
    ):
        """Initialize structured logger.

        Args:
            name: Logger name
            log_dir: Directory for log files (None = no file logging)
            level: Log level
            json_output: Whether to output JSON format
            console_output: Whether to output to console
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.json_output = json_output
        self.console_output = console_output

        # Create logger
        self._logger = logging.getLogger(f"structured.{name}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Clear existing handlers
        self._logger.handlers.clear()

        # JSON file handler
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._file_handler = logging.FileHandler(
                str(self.log_dir / f"{name}.jsonl"),
                encoding="utf-8",
                mode="a",
            )
            self._file_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(self._file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            self._logger.addHandler(console_handler)

        # Context storage (thread-local)
        self._context = threading.local()

        # Performance counters
        self._counters: Dict[str, PerformanceCounter] = {}

    def _get_context(self) -> LogContext:
        """Get current log context."""
        if not hasattr(self._context, "stack"):
            self._context.stack = [LogContext()]
        return self._context.stack[-1]

    def push_context(self, **kwargs):
        """Push a new log context."""
        if not hasattr(self._context, "stack"):
            self._context.stack = [LogContext()]

        current = self._context.stack[-1]
        new_context = LogContext(
            query_id=kwargs.get("query_id", current.query_id),
            session_id=kwargs.get("session_id", current.session_id),
            layer=kwargs.get("layer", current.layer),
            component=kwargs.get("component", current.component),
            user_id=kwargs.get("user_id", current.user_id),
        )
        self._context.stack.append(new_context)

    def pop_context(self):
        """Pop the current log context."""
        if hasattr(self._context, "stack") and len(self._context.stack) > 1:
            self._context.stack.pop()

    def _format_record(self, level: str, message: str, **kwargs) -> str:
        """Format a log record as JSON."""
        context = self._get_context()

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.name,
            "message": message,
        }

        # Add context fields
        if context.query_id:
            record["query_id"] = context.query_id
        if context.session_id:
            record["session_id"] = context.session_id
        if context.layer:
            record["layer"] = context.layer
        if context.component:
            record["component"] = context.component
        if context.user_id:
            record["user_id"] = context.user_id

        # Add extra fields
        for key, value in kwargs.items():
            if key.startswith("_"):
                continue
            record[key] = value

        return json.dumps(record, default=str, ensure_ascii=False)

    def _log(self, level: str, message: str, **kwargs):
        """Internal log method."""
        if self.json_output:
            formatted = self._format_record(level, message, **kwargs)
            if self.console_output:
                print(formatted)
            if self.log_dir:
                self._logger.debug(formatted)
        else:
            log_level = getattr(logging, level.upper(), logging.INFO)
            self._logger.log(log_level, message)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)

    def pipeline(self, message: str, **kwargs):
        """Log pipeline-specific message."""
        self._log("PIPELINE", message, **kwargs)

    # Performance counter methods
    def start_counter(self, name: str, metadata: Optional[Dict] = None):
        """Start a performance counter."""
        self._counters[name] = PerformanceCounter(
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
        )

    def stop_counter(self, name: str) -> float:
        """Stop a performance counter and return elapsed time in ms."""
        if name in self._counters:
            counter = self._counters[name]
            counter.end_time = time.time()
            elapsed = counter.elapsed
            self.info(
                f"Performance: {name}",
                **{
                    "counter_name": name,
                    "elapsed_ms": round(elapsed, 2),
                    "unit": "ms",
                    **counter.metadata,
                },
            )
            return elapsed
        return 0.0

    def get_counter(self, name: str) -> Optional[PerformanceCounter]:
        """Get a performance counter."""
        return self._counters.get(name)

    def reset_counters(self):
        """Reset all performance counters."""
        self._counters.clear()

    # Convenience methods for AARAG layers
    def log_layer_start(self, layer_num: int, layer_name: str, query_id: Optional[str] = None):
        """Log layer start."""
        self.push_context(layer=layer_name)
        self.start_counter(f"layer_{layer_num}")
        self.pipeline(
            f"Layer {layer_num} started: {layer_name}",
            layer_num=layer_num,
            layer_name=layer_name,
            event="layer_start",
            query_id=query_id,
        )

    def log_layer_end(self, layer_num: int, layer_name: str, query_id: Optional[str] = None):
        """Log layer end."""
        elapsed = self.stop_counter(f"layer_{layer_num}")
        self.pipeline(
            f"Layer {layer_num} completed: {layer_name}",
            layer_num=layer_num,
            layer_name=layer_name,
            elapsed_ms=round(elapsed, 2),
            event="layer_end",
            query_id=query_id,
        )
        self.pop_context()

    def log_query_start(self, query: str, query_id: Optional[str] = None):
        """Log query start."""
        if query_id is None:
            query_id = str(uuid.uuid4())[:8]
        self.push_context(query_id=query_id)
        self.start_counter("total_query")
        self.info(
            "Query started",
            event="query_start",
            query=query[:200],
            query_id=query_id,
        )
        return query_id

    def log_query_end(self, query_id: str, answer: str, confidence: float, strategy: str):
        """Log query end."""
        elapsed = self.stop_counter("total_query")
        self.info(
            "Query completed",
            event="query_end",
            query_id=query_id,
            answer_preview=answer[:100],
            confidence=round(confidence, 4),
            strategy=strategy,
            total_ms=round(elapsed, 2),
        )
        self.pop_context()

    def log_decision(self, decision_type: str, value: str, confidence: float, **kwargs):
        """Log a decision point."""
        self.info(
            f"Decision: {decision_type} = {value}",
            event="decision",
            decision_type=decision_type,
            decision_value=value,
            confidence=round(confidence, 4),
            **kwargs,
        )

    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log an error with context."""
        self.error(
            f"Error in {context}: {str(error)}",
            event="error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **kwargs,
        )


# Global structured logger instance
_global_logger: Optional[StructuredLogger] = None


def get_structured_logger(
    name: str = "aarag",
    log_dir: Optional[str] = None,
    **kwargs,
) -> StructuredLogger:
    """Get or create the global structured logger.

    Args:
        name: Logger name
        log_dir: Directory for log files
        **kwargs: Additional arguments to StructuredLogger

    Returns:
        StructuredLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name=name, log_dir=log_dir, **kwargs)
    return _global_logger
