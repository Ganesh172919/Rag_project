"""Ablation — Dedicated ablation study runner."""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ABLATION_CONFIGS = {
    "full": {"disable_router": False, "disable_crag": False, "disable_reflection": False, "disable_graph": False},
    "no_router": {"disable_router": True, "disable_crag": False, "disable_reflection": False, "disable_graph": False},
    "no_crag": {"disable_router": False, "disable_crag": True, "disable_reflection": False, "disable_graph": False},
    "no_reflection": {"disable_router": False, "disable_crag": False, "disable_reflection": True, "disable_graph": False},
    "no_graph": {"disable_router": False, "disable_crag": False, "disable_reflection": False, "disable_graph": True},
    "no_agent": {"disable_router": False, "disable_crag": False, "disable_reflection": False, "disable_graph": False},
}

ABLATION_LABELS = {
    "full": "Full AARAG",
    "no_router": "w/o Router",
    "no_crag": "w/o CRAG",
    "no_reflection": "w/o Self-Reflection",
    "no_graph": "w/o Graph",
    "no_agent": "w/o Agent",
}


def run_ablation_analysis(results_path: str = "results/evaluation_results.json", output_dir: str = "results"):
    """Analyze ablation results and generate comparison tables."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    ablation = results.get("ablation", {})
    if not ablation:
        print("No ablation results found.")
        return

    # Build comparison table
    rows = []
    for config_name, metrics in ablation.items():
        if isinstance(metrics, dict):
            row = {
                "Configuration": ABLATION_LABELS.get(config_name, config_name),
                "EM": metrics.get("exact_match", 0),
                "F1": metrics.get("f1", 0),
                "ROUGE-L": metrics.get("rouge_l", 0),
                "Faithfulness": metrics.get("faithfulness", 0),
                "Latency": metrics.get("avg_latency", 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Compute delta from full system
    if "Full AARAG" in df["Configuration"].values:
        full_row = df[df["Configuration"] == "Full AARAG"].iloc[0]
        df["EM_delta"] = df["EM"] - full_row["EM"]
        df["F1_delta"] = df["F1"] - full_row["F1"]

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "ablation_results.csv", index=False)

    # Print
    print("\nAblation Study Results:")
    print(df.to_string(index=False))

    # Generate LaTeX
    _generate_ablation_latex(df, output_path / "ablation_table.tex")

    return df


def _generate_ablation_latex(df: pd.DataFrame, path: Path):
    """Generate LaTeX table for ablation study."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Ablation Study: Contribution of Each AARAG Component}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{l|cccc}",
        "\\hline",
        "Configuration & EM & F1 & Faithfulness & Latency \\\\",
        "\\hline",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{row['Configuration']} & {row['EM']:.3f} & {row['F1']:.3f} "
            f"& {row['Faithfulness']:.3f} & {row['Latency']:.2f}s \\\\"
        )

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_ablation_analysis()
