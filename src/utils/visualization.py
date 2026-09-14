"""Visualization — Plotting utilities for evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "f1",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot bar chart comparing systems on a metric.

    Args:
        results: Dict of dataset -> {system -> metrics}
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    # Collect data
    systems = set()
    for dataset_results in results.values():
        if isinstance(dataset_results, dict):
            systems.update(dataset_results.keys())

    systems = sorted(systems)
    datasets = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(systems)

    for i, system in enumerate(systems):
        values = []
        for dataset in datasets:
            dataset_result = results.get(dataset, {})
            if isinstance(dataset_result, dict):
                system_metrics = dataset_result.get(system, {})
                if isinstance(system_metrics, dict):
                    values.append(system_metrics.get(metric, 0))
                else:
                    values.append(0)
            else:
                values.append(0)

        offset = (i - len(systems) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=system)

    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation_chart(
    ablation_results: Dict[str, Dict[str, float]],
    metric: str = "f1",
    save_path: Optional[str] = None,
):
    """Plot ablation study results.

    Args:
        ablation_results: Dict of config_name -> metrics
        metric: Metric to plot
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    labels = {
        "full": "Full AARAG",
        "no_router": "w/o Router",
        "no_crag": "w/o CRAG",
        "no_reflection": "w/o Self-Ref",
        "no_graph": "w/o Graph",
    }

    configs = []
    values = []
    colors = []

    for config_name in ["full", "no_router", "no_crag", "no_reflection", "no_graph"]:
        if config_name in ablation_results:
            configs.append(labels.get(config_name, config_name))
            values.append(ablation_results[config_name].get(metric, 0))
            colors.append("#2ecc71" if config_name == "full" else "#e74c3c")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(configs, values, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Ablation Study: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_latency_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """Plot latency comparison across systems."""
    import matplotlib.pyplot as plt

    systems = []
    latencies = []

    # Get first dataset results
    first_dataset = list(results.values())[0]
    if isinstance(first_dataset, dict):
        for system, metrics in first_dataset.items():
            if isinstance(metrics, dict):
                systems.append(system)
                latencies.append(metrics.get("avg_latency", 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db" if s != "AARAG" else "#e74c3c" for s in systems]
    bars = ax.barh(systems, latencies, color=colors, edgecolor="white")

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", ha="left", va="center")

    ax.set_xlabel("Average Latency (seconds)")
    ax.set_title("Latency Comparison")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def generate_all_plots(results_path: str = "results/evaluation_results.json", output_dir: str = "results/figures"):
    """Generate all evaluation plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Metric comparisons
    for metric in ["f1", "exact_match", "faithfulness", "rouge_l"]:
        try:
            plot_metric_comparison(results, metric=metric, save_path=str(output_path / f"{metric}_comparison.png"))
        except Exception as e:
            print(f"Warning: Could not plot {metric}: {e}")

    # Ablation
    if "ablation" in results:
        for metric in ["f1", "faithfulness"]:
            try:
                plot_ablation_chart(results["ablation"], metric=metric,
                                   save_path=str(output_path / f"ablation_{metric}.png"))
            except Exception as e:
                print(f"Warning: Could not plot ablation {metric}: {e}")

    # Latency
    try:
        plot_latency_comparison(results, save_path=str(output_path / "latency_comparison.png"))
    except Exception as e:
        print(f"Warning: Could not plot latency: {e}")

    print(f"Plots saved to {output_path}")


if __name__ == "__main__":
    generate_all_plots()
