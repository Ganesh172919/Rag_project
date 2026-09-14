"""Pipeline Tracer — End-to-end pipeline tracing for AARAG.

Provides OpenTelemetry-compatible tracing with:
- Per-query trace with all layer decisions
- Span-based timing for each pipeline stage
- Visual trace export (JSON for Jaeger/Zipkin)
- Performance analysis per trace
"""

import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger("aarag.utils.pipeline_tracer")


@dataclass
class Span:
    """A single trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, unset

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


@dataclass
class Trace:
    """A complete pipeline trace."""
    trace_id: str
    query: str
    start_time: float
    end_time: float = 0.0
    spans: List[Span] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get total trace duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "query": self.query[:200],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
            "attributes": self.attributes,
        }

    def to_jaeger_format(self) -> Dict[str, Any]:
        """Convert trace to Jaeger-compatible format."""
        return {
            "traceID": self.trace_id,
            "spans": [
                {
                    "traceID": self.trace_id,
                    "spanID": s.span_id,
                    "parentSpanID": s.parent_span_id or "",
                    "operationName": s.name,
                    "startTime": int(s.start_time * 1_000_000),  # microseconds
                    "duration": int(s.duration_ms * 1000),  # microseconds
                    "tags": [
                        {"key": k, "value": v}
                        for k, v in s.attributes.items()
                    ],
                    "logs": [
                        {
                            "timestamp": int(e.get("timestamp", 0) * 1_000_000),
                            "fields": [
                                {"key": k, "value": v}
                                for k, v in e.items() if k != "timestamp"
                            ],
                        }
                        for e in s.events
                    ],
                    "process": {"serviceName": "aarag"},
                }
                for s in self.spans
            ],
        }


class PipelineTracer:
    """End-to-end pipeline tracer for AARAG.

    Traces the full pipeline execution with spans for each layer,
    enabling performance analysis and debugging.

    Example:
        >>> tracer = PipelineTracer(trace_dir="logs/traces/")
        >>> trace = tracer.start_trace("What is ML?")
        >>> with tracer.span("router") as s:
        ...     # Router logic
        ...     s.set_attribute("strategy", "multi_step")
        >>> tracer.end_trace(trace, answer="...", confidence=0.85)
    """

    def __init__(self, trace_dir: Optional[str] = None):
        """Initialize pipeline tracer.

        Args:
            trace_dir: Directory to save trace files (None = in-memory only)
        """
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self._current_trace: Optional[Trace] = None
        self._current_span_stack: List[Span] = []
        self._traces: List[Trace] = []

        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PipelineTracer initialized (trace_dir={trace_dir})")

    def start_trace(self, query: str, **attributes) -> Trace:
        """Start a new trace.

        Args:
            query: The query being traced
            **attributes: Additional trace attributes

        Returns:
            Trace object
        """
        trace_id = str(uuid.uuid4()).replace("-", "")[:16]
        trace = Trace(
            trace_id=trace_id,
            query=query,
            start_time=time.time(),
            attributes=attributes,
        )
        self._current_trace = trace
        self._current_span_stack.clear()

        logger.debug(f"Trace started: {trace_id}")
        return trace

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        **attributes,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name (e.g., "router", "retrieval", "crag")
            parent: Parent span (None = use current top of stack)
            **attributes: Span attributes

        Returns:
            Span object
        """
        if self._current_trace is None:
            logger.warning("No active trace, span will be orphaned")
            trace_id = "orphan"
        else:
            trace_id = self._current_trace.trace_id

        parent_span_id = None
        if parent:
            parent_span_id = parent.span_id
        elif self._current_span_stack:
            parent_span_id = self._current_span_stack[-1].span_id

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()).replace("-", "")[:16],
            parent_span_id=parent_span_id,
            name=name,
            start_time=time.time(),
            attributes=attributes,
        )

        self._current_span_stack.append(span)

        if self._current_trace:
            self._current_trace.spans.append(span)

        logger.debug(f"Span started: {name} (trace={trace_id})")
        return span

    def end_span(self, span: Span, status: str = "ok", **attributes):
        """End a span.

        Args:
            span: The span to end
            status: Span status ("ok", "error")
            **attributes: Final attributes
        """
        span.end_time = time.time()
        span.status = status
        span.attributes.update(attributes)

        if self._current_span_stack and self._current_span_stack[-1].span_id == span.span_id:
            self._current_span_stack.pop()

        logger.debug(
            f"Span ended: {span.name} ({span.duration_ms:.2f}ms, status={status})"
        )

    def add_event(self, span: Span, name: str, **attributes):
        """Add an event to a span.

        Args:
            span: The span to add event to
            name: Event name
            **attributes: Event attributes
        """
        event = {
            "name": name,
            "timestamp": time.time(),
            **attributes,
        }
        span.events.append(event)

    def end_trace(
        self,
        trace: Optional[Trace] = None,
        answer: str = "",
        confidence: float = 0.0,
        strategy: str = "",
    ):
        """End a trace and save it.

        Args:
            trace: The trace to end (None = current trace)
            answer: Final answer
            confidence: Final confidence
            strategy: Strategy used
        """
        if trace is None:
            trace = self._current_trace

        if trace is None:
            logger.warning("No trace to end")
            return

        trace.end_time = time.time()
        trace.attributes.update({
            "answer_preview": answer[:100],
            "confidence": confidence,
            "strategy": strategy,
        })

        # Close any remaining spans
        while self._current_span_stack:
            span = self._current_span_stack.pop()
            if span.end_time == 0:
                span.end_time = time.time()

        self._traces.append(trace)

        # Save trace if directory configured
        if self.trace_dir:
            self._save_trace(trace)

        logger.debug(
            f"Trace ended: {trace.trace_id} ({trace.duration_ms:.2f}ms, "
            f"{len(trace.spans)} spans)"
        )

        if self._current_trace == trace:
            self._current_trace = None

    def _save_trace(self, trace: Trace):
        """Save trace to file."""
        filename = f"trace_{trace.trace_id}.json"
        filepath = self.trace_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)

    def save_all_traces(self, output_path: str):
        """Save all traces to a single file.

        Args:
            output_path: Path to save traces
        """
        data = {
            "trace_count": len(self._traces),
            "traces": [t.to_dict() for t in self._traces],
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(self._traces)} traces to {output_path}")

    def save_jaeger_format(self, output_path: str):
        """Save traces in Jaeger-compatible format.

        Args:
            output_path: Path to save traces
        """
        data = {
            "data": [t.to_jaeger_format() for t in self._traces],
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved Jaeger traces to {output_path}")

    def get_trace_summary(self, trace: Trace) -> Dict[str, Any]:
        """Get a summary of a trace.

        Args:
            trace: The trace to summarize

        Returns:
            Summary dictionary
        """
        span_summary = {}
        for span in trace.spans:
            name = span.name
            if name not in span_summary:
                span_summary[name] = {
                    "count": 0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                }
            span_summary[name]["count"] += 1
            span_summary[name]["total_ms"] += span.duration_ms
            span_summary[name]["max_ms"] = max(
                span_summary[name]["max_ms"], span.duration_ms
            )

        return {
            "trace_id": trace.trace_id,
            "query": trace.query[:100],
            "total_ms": round(trace.duration_ms, 2),
            "span_count": len(trace.spans),
            "spans": span_summary,
            "attributes": trace.attributes,
        }

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all traces."""
        return [self.get_trace_summary(t) for t in self._traces]

    def clear(self):
        """Clear all traces."""
        self._traces.clear()
        self._current_trace = None
        self._current_span_stack.clear()
        logger.info("Traces cleared")
