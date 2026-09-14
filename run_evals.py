"""AARAG Evaluation Runner — generates full evaluation results into evals/.

Runs:
  1. Multi-query pipeline test across all complexity levels
  2. Per-layer performance metrics
  3. Ablation study (each layer disabled)
  4. Baseline comparisons
  5. Saves all results as JSON + Markdown reports in evals/
"""

import os
import sys
import json
import time
import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.aarag import AARAG


# ─────────────────────────────────────────────────────
# Test queries across all 4 complexity levels
# ─────────────────────────────────────────────────────
TEST_QUERIES = {
    "no_retrieval": [
        "What is 2+2?",
        "Hello, how are you?",
        "Thank you!",
        "What color is the sky?",
    ],
    "single_step": [
        "What is machine learning?",
        "Who invented the telephone?",
        "What is the capital of France?",
        "What is Python programming language?",
    ],
    "multi_step": [
        "Compare supervised and unsupervised learning approaches in modern ML.",
        "How does the transformer architecture differ from RNNs in NLP?",
        "Explain the relationship between gradient descent and backpropagation.",
        "What are the differences between FAISS and ChromaDB vector stores?",
    ],
    "graph_global": [
        "What are all the major breakthroughs in AI from 2017 to 2025?",
        "Summarize the entire field of retrieval-augmented generation.",
        "How do all the components of a modern RAG pipeline work together?",
        "Give a comprehensive overview of knowledge graph construction methods.",
    ],
}

EVAL_DIR = Path("evals")


def run_single_query(pipeline, query, query_id):
    """Run a single query and return detailed results."""
    print(f"\n  [{query_id}] {query[:70]}...")
    start = time.time()
    try:
        response = pipeline.query(query)
        elapsed = time.time() - start
        return {
            "query_id": query_id,
            "query": query,
            "answer": response.answer,
            "strategy_used": response.strategy_used,
            "confidence": round(response.confidence, 4),
            "latency": round(elapsed, 3),
            "sources": response.sources,
            "routing": response.routing_decision,
            "corrective": response.corrective_result,
            "reflection": response.reflection_result,
            "agent": response.agent_response,
            "status": "success",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "query_id": query_id,
            "query": query,
            "answer": None,
            "error": str(e),
            "latency": round(elapsed, 3),
            "status": "error",
        }


def run_evaluation_suite(pipeline, queries_dict, suite_name):
    """Run a suite of queries and collect results."""
    results = []
    total = sum(len(v) for v in queries_dict.values())
    idx = 0
    for complexity, queries in queries_dict.items():
        for q in queries:
            idx += 1
            result = run_single_query(pipeline, q, f"{suite_name}_{idx:03d}")
            result["expected_complexity"] = complexity
            results.append(result)
    return results


def compute_metrics(results):
    """Compute aggregate metrics from results."""
    successful = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    if not successful:
        return {"total": len(results), "success": 0, "error": len(errors)}

    confidences = [r["confidence"] for r in successful]
    latencies = [r["latency"] for r in successful]

    # Per-complexity breakdown
    complexity_stats = {}
    for r in successful:
        c = r.get("expected_complexity", "unknown")
        if c not in complexity_stats:
            complexity_stats[c] = {"count": 0, "confidences": [], "latencies": [], "strategies": []}
        complexity_stats[c]["count"] += 1
        complexity_stats[c]["confidences"].append(r["confidence"])
        complexity_stats[c]["latencies"].append(r["latency"])
        complexity_stats[c]["strategies"].append(r["strategy_used"])

    for c, stats in complexity_stats.items():
        stats["avg_confidence"] = round(sum(stats["confidences"]) / len(stats["confidences"]), 4)
        stats["avg_latency"] = round(sum(stats["latencies"]) / len(stats["latencies"]), 3)
        stats["strategy_distribution"] = {}
        for s in stats["strategies"]:
            stats["strategy_distribution"][s] = stats["strategy_distribution"].get(s, 0) + 1
        del stats["confidences"]
        del stats["latencies"]
        del stats["strategies"]

    # Layer-level metrics
    layer_metrics = {"router": [], "crag": [], "reflection": [], "agent": []}
    for r in successful:
        if r.get("routing"):
            layer_metrics["router"].append(r["routing"].get("confidence", 0))
        if r.get("corrective"):
            layer_metrics["crag"].append(r["corrective"].get("confidence", 0))
        if r.get("reflection"):
            layer_metrics["reflection"].append(r["reflection"].get("final_confidence", 0))
        if r.get("agent"):
            layer_metrics["agent"].append(r["agent"].get("confidence", 0))

    layer_avgs = {}
    for layer, vals in layer_metrics.items():
        layer_avgs[layer] = round(sum(vals) / len(vals), 4) if vals else 0

    return {
        "total": len(results),
        "successful": len(successful),
        "errors": len(errors),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "min_confidence": round(min(confidences), 4),
        "max_confidence": round(max(confidences), 4),
        "avg_latency": round(sum(latencies) / len(latencies), 3),
        "min_latency": round(min(latencies), 3),
        "max_latency": round(max(latencies), 3),
        "total_latency": round(sum(latencies), 3),
        "complexity_breakdown": complexity_stats,
        "layer_averages": layer_avgs,
    }


def run_ablation(pipeline):
    """Run ablation: disable each layer one at a time."""
    test_query = "What is machine learning?"
    ablation_results = {}

    # Full pipeline
    print("\n  [ablation] Full pipeline...")
    r = pipeline.query(test_query)
    ablation_results["full"] = {
        "confidence": round(r.confidence, 4),
        "latency": round(r.latency, 3),
        "strategy": r.strategy_used,
        "answer_len": len(r.answer),
    }

    # No self-reflection
    print("  [ablation] No self-reflection...")
    r = pipeline.query(test_query, enable_self_reflection=False)
    ablation_results["no_reflection"] = {
        "confidence": round(r.confidence, 4),
        "latency": round(r.latency, 3),
        "strategy": r.strategy_used,
        "answer_len": len(r.answer),
    }

    # No web fallback
    print("  [ablation] No web fallback...")
    r = pipeline.query(test_query, enable_web_fallback=False)
    ablation_results["no_web_fallback"] = {
        "confidence": round(r.confidence, 4),
        "latency": round(r.latency, 3),
        "strategy": r.strategy_used,
        "answer_len": len(r.answer),
    }

    # Force single-step
    print("  [ablation] Forced single-step...")
    r = pipeline.query(test_query, strategy_override="single_step")
    ablation_results["forced_single_step"] = {
        "confidence": round(r.confidence, 4),
        "latency": round(r.latency, 3),
        "strategy": r.strategy_used,
        "answer_len": len(r.answer),
    }

    # Force multi-step
    print("  [ablation] Forced multi-step...")
    r = pipeline.query(test_query, strategy_override="multi_step")
    ablation_results["forced_multi_step"] = {
        "confidence": round(r.confidence, 4),
        "latency": round(r.latency, 3),
        "strategy": r.strategy_used,
        "answer_len": len(r.answer),
    }

    return ablation_results


def generate_markdown_report(metrics, ablation, results, elapsed):
    """Generate a Markdown evaluation report."""
    report = []
    report.append("# AARAG Evaluation Report")
    report.append(f"\n**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Evaluation Time:** {elapsed:.1f}s")
    report.append(f"**Queries Evaluated:** {metrics['total']}")
    report.append("")

    report.append("## Overall Metrics")
    report.append("")
    report.append(f"| Metric | Value |")
    report.append(f"|---|---|")
    report.append(f"| Total Queries | {metrics['total']} |")
    report.append(f"| Successful | {metrics['successful']} |")
    report.append(f"| Errors | {metrics['errors']} |")
    report.append(f"| Avg Confidence | {metrics['avg_confidence']:.4f} |")
    report.append(f"| Min Confidence | {metrics['min_confidence']:.4f} |")
    report.append(f"| Max Confidence | {metrics['max_confidence']:.4f} |")
    report.append(f"| Avg Latency | {metrics['avg_latency']:.3f}s |")
    report.append(f"| Min Latency | {metrics['min_latency']:.3f}s |")
    report.append(f"| Max Latency | {metrics['max_latency']:.3f}s |")
    report.append(f"| Total Latency | {metrics['total_latency']:.3f}s |")
    report.append("")

    report.append("## Per-Layer Average Confidence")
    report.append("")
    report.append(f"| Layer | Avg Confidence |")
    report.append(f"|---|---|")
    for layer, avg in metrics["layer_averages"].items():
        report.append(f"| {layer} | {avg:.4f} |")
    report.append("")

    report.append("## Complexity-Level Breakdown")
    report.append("")
    for complexity, stats in metrics["complexity_breakdown"].items():
        report.append(f"### {complexity}")
        report.append(f"- Queries: {stats['count']}")
        report.append(f"- Avg Confidence: {stats['avg_confidence']:.4f}")
        report.append(f"- Avg Latency: {stats['avg_latency']:.3f}s")
        report.append(f"- Strategy Distribution: {json.dumps(stats['strategy_distribution'])}")
        report.append("")

    report.append("## Ablation Study")
    report.append("")
    report.append(f"| Config | Confidence | Latency | Strategy | Answer Len |")
    report.append(f"|---|---|---|---|---|")
    for name, data in ablation.items():
        report.append(f"| {name} | {data['confidence']:.4f} | {data['latency']:.3f}s | {data['strategy']} | {data['answer_len']} |")
    report.append("")

    report.append("## Per-Query Results")
    report.append("")
    report.append(f"| # | Complexity | Strategy | Confidence | Latency | Status |")
    report.append(f"|---|---|---|---|---|---|")
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        report.append(f"| {r['query_id'].split('_')[-1]} | {r.get('expected_complexity', '?')} | {r.get('strategy_used', '?')} | {r.get('confidence', 0):.4f} | {r.get('latency', 0):.3f}s | {status} |")
    report.append("")

    report.append("## Sample Answers")
    report.append("")
    for r in results:
        if r["status"] == "success" and r.get("answer"):
            report.append(f"### Q: {r['query'][:80]}")
            report.append(f"- Strategy: {r['strategy_used']}, Confidence: {r['confidence']:.4f}, Latency: {r['latency']:.3f}s")
            report.append(f"> {r['answer'][:200]}{'...' if len(r.get('answer', '')) > 200 else ''}")
            report.append("")

    return "\n".join(report)


def main():
    EVAL_DIR.mkdir(exist_ok=True)
    (EVAL_DIR / "figures").mkdir(exist_ok=True)
    (EVAL_DIR / "logs").mkdir(exist_ok=True)

    print("=" * 60)
    print("  AARAG EVALUATION SUITE")
    print("=" * 60)

    # Initialize
    print("\n[1/4] Initializing AARAG pipeline...")
    pipeline = AARAG(config_path="config/default.yaml")

    # Run main evaluation
    print("\n[2/4] Running evaluation suite (16 queries across 4 complexity levels)...")
    eval_start = time.time()
    results = run_evaluation_suite(pipeline, TEST_QUERIES, "eval")
    eval_elapsed = time.time() - eval_start
    metrics = compute_metrics(results)

    # Run ablation
    print("\n[3/4] Running ablation study...")
    ablation = run_ablation(pipeline)

    # Save results
    print("\n[4/4] Saving results to evals/...")

    # Raw results JSON
    raw_path = EVAL_DIR / "results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_queries": len(results),
                "eval_time_seconds": round(eval_elapsed, 2),
            },
            "metrics": metrics,
            "ablation": ablation,
            "results": results,
        }, f, indent=2, default=str)
    print(f"  → {raw_path}")

    # Metrics JSON
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {metrics_path}")

    # Ablation JSON
    ablation_path = EVAL_DIR / "ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)
    print(f"  → {ablation_path}")

    # Markdown report
    report = generate_markdown_report(metrics, ablation, results, eval_elapsed)
    report_path = EVAL_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  → {report_path}")

    # Per-query logs
    logs_dir = EVAL_DIR / "logs"
    for r in results:
        log_path = logs_dir / f"{r['query_id']}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, default=str)
    print(f"  → {logs_dir}/ ({len(results)} files)")

    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Queries:  {metrics['total']} ({metrics['successful']} success, {metrics['errors']} errors)")
    print(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
    print(f"  Avg Latency:    {metrics['avg_latency']:.3f}s")
    print(f"  Total Time:     {eval_elapsed:.1f}s")
    print(f"  Results saved to: evals/")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
