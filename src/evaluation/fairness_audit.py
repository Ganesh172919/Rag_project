"""Fairness Audit — Bias detection and consistency testing for AARAG.

Provides fairness auditing including:
- Consistency testing (paraphrase stability)
- Query type bias detection
- Answer quality variance analysis
- Confidence calibration analysis
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("aarag.evaluation.fairness_audit")


@dataclass
class ConsistencyResult:
    """Result of consistency testing."""
    original_query: str
    paraphrases: List[str]
    original_answer: str
    paraphrase_answers: List[str]
    answer_similarity: float
    strategy_match: bool
    confidence_variance: float


@dataclass
class BiasResult:
    """Result of bias detection."""
    query_type: str
    sample_count: int
    avg_f1: float
    avg_confidence: float
    avg_latency: float
    strategy_distribution: Dict[str, int]


@dataclass
class FairnessReport:
    """Complete fairness audit report."""
    consistency_score: float
    consistency_results: List[ConsistencyResult]
    bias_results: List[BiasResult]
    confidence_calibration: Dict[str, float]
    recommendations: List[str]


class FairnessAuditor:
    """Audits AARAG for fairness and consistency.

    Tests:
    1. Consistency: Same answer for paraphrased queries
    2. Bias: Performance across query types
    3. Calibration: Confidence vs actual accuracy
    4. Variance: Answer quality stability

    Example:
        >>> auditor = FairnessAuditor()
        >>> report = auditor.audit(pipeline, test_data)
        >>> auditor.print_report(report)
    """

    def __init__(self):
        """Initialize fairness auditor."""
        logger.info("FairnessAuditor initialized")

    def test_consistency(
        self,
        pipeline,
        queries: List[str],
        paraphrases: List[List[str]],
    ) -> List[ConsistencyResult]:
        """Test consistency across paraphrased queries.

        Args:
            pipeline: AARAG pipeline instance
            queries: Original queries
            paraphrases: List of paraphrase lists (one per original query)

        Returns:
            List of ConsistencyResult
        """
        results = []

        for query, paraphrase_list in zip(queries, paraphrases):
            # Get original response
            original_response = pipeline.query(query)

            # Get paraphrase responses
            paraphrase_answers = []
            paraphrase_strategies = []
            paraphrase_confidences = []

            for para in paraphrase_list:
                try:
                    para_response = pipeline.query(para)
                    paraphrase_answers.append(para_response.answer)
                    paraphrase_strategies.append(para_response.strategy_used)
                    paraphrase_confidences.append(para_response.confidence)
                except Exception as e:
                    logger.warning(f"Error on paraphrase: {e}")
                    paraphrase_answers.append("")
                    paraphrase_strategies.append("error")
                    paraphrase_confidences.append(0.0)

            # Compute answer similarity
            answer_similarities = []
            for para_answer in paraphrase_answers:
                sim = self._compute_similarity(original_response.answer, para_answer)
                answer_similarities.append(sim)

            avg_similarity = (
                statistics.mean(answer_similarities) if answer_similarities else 0.0
            )

            # Check strategy match
            strategy_matches = [
                s == original_response.strategy_used for s in paraphrase_strategies
            ]
            strategy_match = all(strategy_matches) if strategy_matches else False

            # Confidence variance
            all_confidences = [original_response.confidence] + paraphrase_confidences
            conf_variance = (
                statistics.variance(all_confidences) if len(all_confidences) > 1 else 0.0
            )

            results.append(ConsistencyResult(
                original_query=query,
                paraphrases=paraphrase_list,
                original_answer=original_response.answer,
                paraphrase_answers=paraphrase_answers,
                answer_similarity=avg_similarity,
                strategy_match=strategy_match,
                confidence_variance=conf_variance,
            ))

        logger.info(f"Consistency test: {len(results)} query groups tested")
        return results

    def test_bias(
        self,
        pipeline,
        test_data: Dict[str, List[Dict]],
    ) -> List[BiasResult]:
        """Test for bias across query types.

        Args:
            pipeline: AARAG pipeline instance
            test_data: Dict of query_type -> list of {question, answer}

        Returns:
            List of BiasResult per query type
        """
        results = []

        for query_type, items in test_data.items():
            f1_scores = []
            confidences = []
            latencies = []
            strategies = []

            for item in items[:50]:  # Limit to 50 per type
                try:
                    response = pipeline.query(item["question"])

                    # Compute F1
                    f1 = self._compute_f1(response.answer, item.get("answer", ""))
                    f1_scores.append(f1)
                    confidences.append(response.confidence)
                    latencies.append(response.latency)
                    strategies.append(response.strategy_used)
                except Exception as e:
                    logger.warning(f"Error on {query_type} query: {e}")

            if f1_scores:
                strategy_dist = {}
                for s in strategies:
                    strategy_dist[s] = strategy_dist.get(s, 0) + 1

                results.append(BiasResult(
                    query_type=query_type,
                    sample_count=len(f1_scores),
                    avg_f1=statistics.mean(f1_scores),
                    avg_confidence=statistics.mean(confidences),
                    avg_latency=statistics.mean(latencies),
                    strategy_distribution=strategy_dist,
                ))

        logger.info(f"Bias test: {len(results)} query types analyzed")
        return results

    def audit(
        self,
        pipeline,
        test_data: Dict[str, List[Dict]],
        paraphrase_data: Optional[Dict[str, List[str]]] = None,
    ) -> FairnessReport:
        """Run full fairness audit.

        Args:
            pipeline: AARAG pipeline instance
            test_data: Dict of query_type -> list of {question, answer}
            paraphrase_data: Optional dict of query -> list of paraphrases

        Returns:
            FairnessReport with all audit results
        """
        logger.info("Starting fairness audit...")

        # Consistency testing
        consistency_results = []
        if paraphrase_data:
            queries = list(paraphrase_data.keys())[:20]  # Limit to 20
            paraphrases = [paraphrase_data[q] for q in queries]
            consistency_results = self.test_consistency(pipeline, queries, paraphrases)

        consistency_score = (
            statistics.mean([r.answer_similarity for r in consistency_results])
            if consistency_results else 0.0
        )

        # Bias testing
        bias_results = self.test_bias(pipeline, test_data)

        # Confidence calibration
        calibration = self._analyze_calibration(bias_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            consistency_score, bias_results, calibration
        )

        report = FairnessReport(
            consistency_score=consistency_score,
            consistency_results=consistency_results,
            bias_results=bias_results,
            confidence_calibration=calibration,
            recommendations=recommendations,
        )

        logger.info("Fairness audit complete")
        return report

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using token overlap."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        overlap = len(tokens1 & tokens2)
        total = max(len(tokens1), len(tokens2))

        return overlap / total

    def _compute_f1(self, predicted: str, reference: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _analyze_calibration(self, bias_results: List[BiasResult]) -> Dict[str, float]:
        """Analyze confidence calibration."""
        calibration = {}

        for result in bias_results:
            # Calibration gap: confidence - accuracy
            gap = result.avg_confidence - result.avg_f1
            calibration[result.query_type] = round(gap, 4)

        return calibration

    def _generate_recommendations(
        self,
        consistency_score: float,
        bias_results: List[BiasResult],
        calibration: Dict[str, float],
    ) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []

        # Consistency recommendations
        if consistency_score < 0.7:
            recommendations.append(
                "Low consistency score. Consider adding query normalization "
                "or semantic caching to improve paraphrase stability."
            )

        # Bias recommendations
        if bias_results:
            f1_values = [r.avg_f1 for r in bias_results]
            if max(f1_values) - min(f1_values) > 0.2:
                worst_type = min(bias_results, key=lambda r: r.avg_f1)
                recommendations.append(
                    f"Performance varies significantly across query types. "
                    f"'{worst_type.query_type}' has lowest F1 ({worst_type.avg_f1:.3f}). "
                    f"Consider adding training data for this type."
                )

        # Calibration recommendations
        for query_type, gap in calibration.items():
            if abs(gap) > 0.15:
                direction = "overconfident" if gap > 0 else "underconfident"
                recommendations.append(
                    f"System is {direction} for '{query_type}' queries "
                    f"(gap: {gap:+.3f}). Consider recalibrating confidence scores."
                )

        if not recommendations:
            recommendations.append("No significant fairness issues detected.")

        return recommendations

    def save_report(self, report: FairnessReport, output_path: str):
        """Save fairness audit report."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": {
                "consistency_score": round(report.consistency_score, 4),
                "bias_results": [
                    {
                        "query_type": r.query_type,
                        "sample_count": r.sample_count,
                        "avg_f1": round(r.avg_f1, 4),
                        "avg_confidence": round(r.avg_confidence, 4),
                        "avg_latency": round(r.avg_latency, 4),
                        "strategy_distribution": r.strategy_distribution,
                    }
                    for r in report.bias_results
                ],
                "confidence_calibration": report.confidence_calibration,
                "recommendations": report.recommendations,
            },
            "consistency_results": [
                {
                    "original_query": r.original_query,
                    "answer_similarity": round(r.answer_similarity, 4),
                    "strategy_match": r.strategy_match,
                    "confidence_variance": round(r.confidence_variance, 4),
                }
                for r in report.consistency_results
            ],
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Fairness report saved to {output_path}")

    def print_report(self, report: FairnessReport):
        """Print fairness audit report to console."""
        print("\n" + "=" * 60)
        print("  FAIRNESS AUDIT REPORT")
        print("=" * 60)

        print(f"\n  Consistency Score: {report.consistency_score:.4f}")

        print("\n  Bias Analysis:")
        print(f"  {'Query Type':<20} {'Samples':>8} {'F1':>8} {'Conf':>8} {'Latency':>8}")
        print("  " + "-" * 56)
        for r in report.bias_results:
            print(
                f"  {r.query_type:<20} {r.sample_count:>8} "
                f"{r.avg_f1:>8.3f} {r.avg_confidence:>8.3f} {r.avg_latency:>7.3f}s"
            )

        print("\n  Confidence Calibration (gap = confidence - accuracy):")
        for query_type, gap in report.confidence_calibration.items():
            status = "✓" if abs(gap) < 0.1 else "⚠"
            print(f"    {status} {query_type:<20} {gap:+.4f}")

        print("\n  Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"    {i}. {rec}")

        print("=" * 60)
