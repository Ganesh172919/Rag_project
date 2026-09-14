"""Error Analysis — Categorizes and analyzes evaluation errors.

Provides detailed error analysis including:
- Error categorization (wrong retrieval, hallucination, incomplete, etc.)
- Per-error-type analysis
- Error correlation with query complexity
- Error pattern identification
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter

logger = logging.getLogger("aarag.evaluation.error_analysis")


@dataclass
class ErrorCase:
    """A single error case."""
    query: str
    predicted: str
    reference: str
    error_type: str
    complexity: str
    confidence: float
    latency: float
    details: str = ""


@dataclass
class ErrorSummary:
    """Summary of error analysis."""
    total_errors: int
    error_rate: float
    error_types: Dict[str, int]
    error_by_complexity: Dict[str, int]
    avg_confidence_by_type: Dict[str, float]
    common_patterns: List[Tuple[str, int]]


class ErrorAnalyzer:
    """Analyzes and categorizes evaluation errors.

    Categorizes errors into:
    - wrong_retrieval: Retrieved irrelevant documents
    - hallucination: Generated unsupported claims
    - incomplete: Missed part of the question
    - wrong_reasoning: Incorrect logical inference
    - format_error: Formatting or truncation issues
    - no_answer: Failed to generate an answer

    Example:
        >>> analyzer = ErrorAnalyzer()
        >>> errors = analyzer.analyze(predictions, references, questions)
        >>> summary = analyzer.summarize(errors)
        >>> analyzer.save_report(errors, summary, "results/error_analysis.json")
    """

    ERROR_TYPES = [
        "wrong_retrieval",
        "hallucination",
        "incomplete",
        "wrong_reasoning",
        "format_error",
        "no_answer",
    ]

    def __init__(self):
        """Initialize error analyzer."""
        logger.info("ErrorAnalyzer initialized")

    def _categorize_error(
        self,
        predicted: str,
        reference: str,
        question: str,
    ) -> Tuple[str, str]:
        """Categorize a single error.

        Args:
            predicted: Predicted answer
            reference: Reference answer
            question: Original question

        Returns:
            Tuple of (error_type, details)
        """
        pred_lower = predicted.lower().strip()
        ref_lower = reference.lower().strip()

        # No answer
        if not pred_lower or pred_lower in ["", "n/a", "none", "no answer"]:
            return "no_answer", "Empty or null prediction"

        # Format error (truncation, encoding issues)
        if len(predicted) > len(reference) * 5:
            return "format_error", "Answer is significantly too long"

        if "..." in predicted and predicted.endswith("..."):
            return "format_error", "Answer appears truncated"

        # Check for hallucination (answer contains info not in reference)
        ref_words = set(ref_lower.split())
        pred_words = set(pred_lower.split())

        # Words in prediction not in reference (potential hallucination)
        novel_words = pred_words - ref_words
        novel_ratio = len(novel_words) / len(pred_words) if pred_words else 0

        if novel_ratio > 0.7:
            return "hallucination", f"High novel content ratio: {novel_ratio:.2f}"

        # Check for incomplete (reference contains info not in prediction)
        missing_words = ref_words - pred_words
        missing_ratio = len(missing_words) / len(ref_words) if ref_words else 0

        if missing_ratio > 0.6:
            return "incomplete", f"Missing {missing_ratio:.0%} of reference content"

        # Check for wrong reasoning (some overlap but wrong answer)
        overlap = len(ref_words & pred_words)
        overlap_ratio = overlap / max(len(ref_words), len(pred_words)) if ref_words else 0

        if 0.2 < overlap_ratio < 0.5:
            return "wrong_reasoning", f"Partial overlap ({overlap_ratio:.2f}) suggests wrong reasoning"

        # Default: wrong retrieval (low overlap)
        return "wrong_retrieval", f"Low content overlap: {overlap_ratio:.2f}"

    def analyze(
        self,
        predictions: List[str],
        references: List[str],
        questions: List[str],
        complexities: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
        latencies: Optional[List[float]] = None,
    ) -> List[ErrorCase]:
        """Analyze errors in predictions.

        Args:
            predictions: List of predicted answers
            references: List of reference answers
            questions: List of questions
            complexities: Optional list of query complexities
            confidences: Optional list of confidence scores
            latencies: Optional list of latencies

        Returns:
            List of ErrorCase for incorrect predictions
        """
        if complexities is None:
            complexities = ["unknown"] * len(predictions)
        if confidences is None:
            confidences = [0.0] * len(predictions)
        if latencies is None:
            latencies = [0.0] * len(predictions)

        errors = []

        for i, (pred, ref, q) in enumerate(zip(predictions, references, questions)):
            # Check if prediction is correct
            is_correct = self._is_correct(pred, ref)

            if not is_correct:
                error_type, details = self._categorize_error(pred, ref, q)

                errors.append(ErrorCase(
                    query=q,
                    predicted=pred,
                    reference=ref,
                    error_type=error_type,
                    complexity=complexities[i],
                    confidence=confidences[i],
                    latency=latencies[i],
                    details=details,
                ))

        logger.info(
            f"Error analysis: {len(errors)} errors out of {len(predictions)} predictions"
        )

        return errors

    def _is_correct(self, predicted: str, reference: str) -> bool:
        """Check if prediction matches reference.

        Uses token-level F1 with threshold.
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())

        if not ref_tokens:
            return not pred_tokens

        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(ref_tokens)

        if precision + recall == 0:
            return False

        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5

    def summarize(self, errors: List[ErrorCase]) -> ErrorSummary:
        """Summarize error analysis.

        Args:
            errors: List of ErrorCase

        Returns:
            ErrorSummary with aggregated statistics
        """
        if not errors:
            return ErrorSummary(
                total_errors=0,
                error_rate=0.0,
                error_types={},
                error_by_complexity={},
                avg_confidence_by_type={},
                common_patterns=[],
            )

        # Error type distribution
        error_types = Counter(e.error_type for e in errors)

        # Error by complexity
        error_by_complexity = Counter(e.complexity for e in errors)

        # Average confidence by error type
        conf_by_type = {}
        for error_type in self.ERROR_TYPES:
            type_errors = [e for e in errors if e.error_type == error_type]
            if type_errors:
                conf_by_type[error_type] = sum(e.confidence for e in type_errors) / len(type_errors)

        # Common patterns (extract from error details)
        patterns = Counter()
        for e in errors:
            if e.details:
                # Extract key phrases
                key_phrase = e.details.split(":")[0] if ":" in e.details else e.details
                patterns[key_phrase] += 1

        return ErrorSummary(
            total_errors=len(errors),
            error_rate=len(errors) / max(1, len(errors)),  # Will be adjusted by caller
            error_types=dict(error_types),
            error_by_complexity=dict(error_by_complexity),
            avg_confidence_by_type=conf_by_type,
            common_patterns=patterns.most_common(10),
        )

    def save_report(
        self,
        errors: List[ErrorCase],
        summary: ErrorSummary,
        output_path: str,
    ):
        """Save error analysis report.

        Args:
            errors: List of ErrorCase
            summary: ErrorSummary
            output_path: Path to save report
        """
        report = {
            "summary": {
                "total_errors": summary.total_errors,
                "error_rate": round(summary.error_rate, 4),
                "error_types": summary.error_types,
                "error_by_complexity": summary.error_by_complexity,
                "avg_confidence_by_type": {
                    k: round(v, 4) for k, v in summary.avg_confidence_by_type.items()
                },
                "common_patterns": [
                    {"pattern": p, "count": c} for p, c in summary.common_patterns
                ],
            },
            "errors": [
                {
                    "query": e.query,
                    "predicted": e.predicted[:200],
                    "reference": e.reference[:200],
                    "error_type": e.error_type,
                    "complexity": e.complexity,
                    "confidence": round(e.confidence, 4),
                    "latency": round(e.latency, 4),
                    "details": e.details,
                }
                for e in errors
            ],
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Error analysis report saved to {output_path}")

    def print_summary(self, summary: ErrorSummary):
        """Print error summary to console."""
        print("\n" + "=" * 60)
        print("  ERROR ANALYSIS")
        print("=" * 60)
        print(f"  Total errors: {summary.total_errors}")
        print()

        print("  Error Types:")
        for error_type, count in sorted(summary.error_types.items(), key=lambda x: -x[1]):
            pct = count / summary.total_errors * 100 if summary.total_errors else 0
            print(f"    {error_type:<20} {count:>4} ({pct:>5.1f}%)")

        print()
        print("  Error by Complexity:")
        for complexity, count in sorted(summary.error_by_complexity.items()):
            print(f"    {complexity:<20} {count:>4}")

        print()
        print("  Avg Confidence by Error Type:")
        for error_type, conf in summary.avg_confidence_by_type.items():
            print(f"    {error_type:<20} {conf:.4f}")

        print()
        print("  Common Patterns:")
        for pattern, count in summary.common_patterns[:5]:
            print(f"    {pattern:<40} {count:>4}")
        print("=" * 60)
