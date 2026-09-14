"""Evaluate — Main evaluation script for AARAG."""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics
from .baselines import get_all_baselines, BaselineResponse


def evaluate_system(
    system,
    eval_data: List[Dict],
    system_name: str = "System",
    max_samples: int = 1000,
) -> Dict:
    """Evaluate a system on a dataset.

    Args:
        system: System with .answer(query) method
        eval_data: List of dicts with 'question' and 'answer' keys
        system_name: Name for logging
        max_samples: Max samples to evaluate

    Returns:
        Dict with metrics and per-sample results
    """
    predictions = []
    references = []
    questions = []
    latencies = []

    samples = eval_data[:max_samples]

    print(f"\nEvaluating {system_name} on {len(samples)} samples...")

    for i, item in enumerate(samples):
        query = item["question"]
        reference = item["answer"]

        try:
            if hasattr(system, "query"):
                # AARAG system
                response = system.query(query)
                prediction = response.answer
                latency = response.latency
            elif hasattr(system, "answer"):
                # Baseline system
                response = system.answer(query)
                prediction = response.answer
                latency = response.latency
            else:
                prediction = str(system)
                latency = 0.0
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            prediction = ""
            latency = 0.0

        predictions.append(prediction)
        references.append(reference)
        questions.append(query)
        latencies.append(latency)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

    # Compute metrics
    metrics = compute_all_metrics(predictions, references, questions)
    metrics["avg_latency"] = np.mean(latencies)
    metrics["total_samples"] = len(samples)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "questions": questions,
        "latencies": latencies,
    }


def run_full_evaluation(
    aarag=None,
    eval_data: Optional[Dict[str, List[Dict]]] = None,
    eval_dir: str = "data/processed",
    output_dir: str = "results",
    max_samples: int = 500,
    run_ablation: bool = True,
) -> Dict:
    """Run full evaluation on all datasets and baselines.

    Args:
        aarag: AARAG system instance
        eval_data: Dict of dataset_name -> evaluation items
        eval_dir: Directory with evaluation data files
        output_dir: Directory to save results
        max_samples: Max samples per dataset
        run_ablation: Whether to run ablation studies

    Returns:
        Dict with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load evaluation data if not provided
    if eval_data is None:
        eval_data = {}
        eval_path = Path(eval_dir)
        for f in eval_path.glob("eval_*.json"):
            name = f.stem.replace("eval_", "")
            with open(f, "r", encoding="utf-8") as fh:
                eval_data[name] = json.load(fh)

    if not eval_data:
        print("No evaluation data found. Run data_preparation.py first.")
        return {}

    all_results = {}

    # Get baselines
    baselines = {}
    if aarag:
        baselines = get_all_baselines(
            vector_store=aarag.vector_store,
            router=aarag.router,
            crag=aarag.crag,
            self_reflection=aarag.self_reflection,
            llm_fn=aarag._get_llm_fn(),
        )

    # Evaluate on each dataset
    for dataset_name, data in eval_data.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({len(data)} samples)")
        print(f"{'='*60}")

        dataset_results = {}

        # Evaluate AARAG
        if aarag:
            aarag_result = evaluate_system(aarag, data, "AARAG", max_samples)
            dataset_results["AARAG"] = aarag_result["metrics"]

        # Evaluate baselines
        for baseline_name, baseline in baselines.items():
            try:
                baseline_result = evaluate_system(baseline, data, baseline_name, max_samples)
                dataset_results[baseline_name] = baseline_result["metrics"]
            except Exception as e:
                print(f"  Error evaluating {baseline_name}: {e}")

        all_results[dataset_name] = dataset_results

    # Run ablation studies
    if run_ablation and aarag:
        print(f"\n{'='*60}")
        print("Ablation Studies")
        print(f"{'='*60}")
        ablation_results = run_ablation_studies(aarag, eval_data, max_samples)
        all_results["ablation"] = ablation_results

    # Save results
    save_results(all_results, output_path)

    # Print summary
    print_summary(all_results)

    return all_results


def run_ablation_studies(
    aarag,
    eval_data: Dict[str, List[Dict]],
    max_samples: int = 200,
) -> Dict:
    """Run ablation studies on AARAG components."""
    ablation_results = {}

    # Use first dataset for ablation
    first_dataset = list(eval_data.values())[0]
    data = first_dataset[:max_samples]

    # Full AARAG
    print("\nAblation: Full AARAG")
    full_result = evaluate_system(aarag, data, "Full AARAG", max_samples)
    ablation_results["full"] = full_result["metrics"]

    # A1: Without router (always multi-step)
    print("\nAblation: Without Router")
    no_router_result = evaluate_system(
        _AARAGAblation(aarag, disable_router=True), data, "w/o Router", max_samples
    )
    ablation_results["no_router"] = no_router_result["metrics"]

    # A2: Without CRAG
    print("\nAblation: Without CRAG")
    no_crag_result = evaluate_system(
        _AARAGAblation(aarag, disable_crag=True), data, "w/o CRAG", max_samples
    )
    ablation_results["no_crag"] = no_crag_result["metrics"]

    # A3: Without self-reflection
    print("\nAblation: Without Self-Reflection")
    no_reflect_result = evaluate_system(
        _AARAGAblation(aarag, disable_reflection=True), data, "w/o Self-Ref", max_samples
    )
    ablation_results["no_reflection"] = no_reflect_result["metrics"]

    # A4: Without graph
    print("\nAblation: Without Graph")
    no_graph_result = evaluate_system(
        _AARAGAblation(aarag, disable_graph=True), data, "w/o Graph", max_samples
    )
    ablation_results["no_graph"] = no_graph_result["metrics"]

    return ablation_results


class _AARAGAblation:
    """Wrapper to disable specific AARAG components for ablation."""

    def __init__(self, aarag, disable_router=False, disable_crag=False,
                 disable_reflection=False, disable_graph=False):
        self.aarag = aarag
        self.disable_router = disable_router
        self.disable_crag = disable_crag
        self.disable_reflection = disable_reflection
        self.disable_graph = disable_graph

    def query(self, question: str, **kwargs):
        return self.aarag.query(
            question,
            enable_web_fallback=not self.disable_crag,
            enable_self_reflection=not self.disable_reflection,
            strategy_override="multi_step" if self.disable_router else None,
        )


def save_results(results: Dict, output_path: Path):
    """Save evaluation results to files."""
    # Save as JSON
    serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, dict) and "metrics" in v:
                    serializable[key][k] = v["metrics"]
                else:
                    serializable[key][k] = v

    with open(output_path / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Save as CSV table
    rows = []
    for dataset_name, dataset_results in results.items():
        if dataset_name == "ablation":
            continue
        for system_name, metrics in dataset_results.items():
            if isinstance(metrics, dict):
                row = {"dataset": dataset_name, "system": system_name}
                row.update(metrics)
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path / "evaluation_results.csv", index=False)
        print(f"\nResults saved to {output_path}")

    # Save LaTeX table
    _save_latex_table(results, output_path / "results_table.tex")


def _save_latex_table(results: Dict, path: Path):
    """Save results as a LaTeX table."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{AARAG Evaluation Results}",
        "\\label{tab:results}",
        "\\begin{tabular}{l|ccccc}",
        "\\hline",
        "System & EM & F1 & ROUGE-L & Faithfulness & Latency \\\\",
        "\\hline",
    ]

    for dataset_name, dataset_results in results.items():
        if dataset_name == "ablation":
            continue
        for system_name, metrics in dataset_results.items():
            if isinstance(metrics, dict):
                em = metrics.get("exact_match", 0)
                f1 = metrics.get("f1", 0)
                rl = metrics.get("rouge_l", 0)
                faith = metrics.get("faithfulness", 0)
                lat = metrics.get("avg_latency", 0)
                lines.append(f"{system_name} & {em:.3f} & {f1:.3f} & {rl:.3f} & {faith:.3f} & {lat:.2f}s \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(results: Dict):
    """Print a summary of evaluation results."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    for dataset_name, dataset_results in results.items():
        print(f"\n--- {dataset_name.upper()} ---")
        print(f"{'System':<20} {'EM':>8} {'F1':>8} {'ROUGE-L':>8} {'Faith':>8} {'Latency':>8}")
        print("-" * 72)

        for system_name, metrics in dataset_results.items():
            if isinstance(metrics, dict):
                em = metrics.get("exact_match", 0)
                f1 = metrics.get("f1", 0)
                rl = metrics.get("rouge_l", 0)
                faith = metrics.get("faithfulness", 0)
                lat = metrics.get("avg_latency", 0)
                print(f"{system_name:<20} {em:>8.3f} {f1:>8.3f} {rl:>8.3f} {faith:>8.3f} {lat:>7.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AARAG system")
    parser.add_argument("--eval-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    run_full_evaluation(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        run_ablation=args.ablation or args.full,
    )
