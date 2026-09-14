"""Generate evaluation figures from AARAG results."""

import json
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("evals")
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(EVAL_DIR / "metrics.json") as f:
    metrics = json.load(f)
with open(EVAL_DIR / "ablation.json") as f:
    ablation = json.load(f)
with open(EVAL_DIR / "results.json") as f:
    full_data = json.load(f)
    results = full_data["results"]

# ──────────────────────────────────────────────
# Figure 1: Per-Query Confidence Bar Chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
queries = [r["query_id"].split("_")[-1] for r in results]
confidences = [r["confidence"] for r in results]
colors = []
for r in results:
    c = r.get("expected_complexity", "")
    if c == "no_retrieval": colors.append("#3498db")
    elif c == "single_step": colors.append("#2ecc71")
    elif c == "multi_step": colors.append("#e67e22")
    elif c == "graph_global": colors.append("#9b59b6")
    else: colors.append("#95a5a6")

bars = ax.bar(queries, confidences, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Query ID", fontsize=11)
ax.set_ylabel("Confidence", fontsize=11)
ax.set_title("AARAG: Per-Query Confidence Across All Complexity Levels", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.axhline(y=metrics["avg_confidence"], color="red", linestyle="--", alpha=0.7, label=f"Avg: {metrics['avg_confidence']:.4f}")
ax.legend(fontsize=10)

# Legend for complexity
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#3498db", label="No Retrieval"),
    Patch(facecolor="#2ecc71", label="Single-Step"),
    Patch(facecolor="#e67e22", label="Multi-Step"),
    Patch(facecolor="#9b59b6", label="Graph-Global"),
]
ax.legend(handles=legend_elements + [ax.get_lines()[0]], loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "confidence_per_query.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: confidence_per_query.png")

# ──────────────────────────────────────────────
# Figure 2: Latency per Query
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
latencies = [r["latency"] for r in results]
bars = ax.bar(queries, latencies, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Query ID", fontsize=11)
ax.set_ylabel("Latency (seconds)", fontsize=11)
ax.set_title("AARAG: Per-Query Latency", fontsize=13, fontweight="bold")
ax.axhline(y=metrics["avg_latency"], color="red", linestyle="--", alpha=0.7, label=f"Avg: {metrics['avg_latency']:.3f}s")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "latency_per_query.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: latency_per_query.png")

# ──────────────────────────────────────────────
# Figure 3: Complexity Level Comparison (Grouped)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

complexities = list(metrics["complexity_breakdown"].keys())
avg_confs = [metrics["complexity_breakdown"][c]["avg_confidence"] for c in complexities]
avg_lats = [metrics["complexity_breakdown"][c]["avg_latency"] for c in complexities]
bar_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

axes[0].bar(complexities, avg_confs, color=bar_colors, edgecolor="white")
axes[0].set_ylabel("Avg Confidence", fontsize=11)
axes[0].set_title("Average Confidence by Complexity Level", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 1.0)
for i, v in enumerate(avg_confs):
    axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

axes[1].bar(complexities, avg_lats, color=bar_colors, edgecolor="white")
axes[1].set_ylabel("Avg Latency (seconds)", fontsize=11)
axes[1].set_title("Average Latency by Complexity Level", fontsize=12, fontweight="bold")
for i, v in enumerate(avg_lats):
    axes[1].text(i, v + 0.1, f"{v:.3f}s", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "complexity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: complexity_comparison.png")

# ──────────────────────────────────────────────
# Figure 4: Per-Layer Confidence Radar Chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
layers = list(metrics["layer_averages"].keys())
values = list(metrics["layer_averages"].values())
values += values[:1]  # Close the polygon

angles = np.linspace(0, 2 * np.pi, len(layers), endpoint=False).tolist()
angles += angles[:1]

ax.fill(angles, values, color="#3498db", alpha=0.25)
ax.plot(angles, values, color="#3498db", linewidth=2, marker="o", markersize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([l.upper() for l in layers], fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_title("AARAG: Per-Layer Average Confidence", fontsize=13, fontweight="bold", y=1.08)

for angle, value in zip(angles[:-1], values[:-1]):
    ax.annotate(f"{value:.4f}", xy=(angle, value), fontsize=10, ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "layer_confidence_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: layer_confidence_radar.png")

# ──────────────────────────────────────────────
# Figure 5: Ablation Study Comparison
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ablation_names = list(ablation.keys())
ablation_confs = [ablation[n]["confidence"] for n in ablation_names]
ablation_lats = [ablation[n]["latency"] for n in ablation_names]
ablation_colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]

axes[0].barh(ablation_names, ablation_confs, color=ablation_colors, edgecolor="white")
axes[0].set_xlabel("Confidence", fontsize=11)
axes[0].set_title("Ablation: Confidence by Configuration", fontsize=12, fontweight="bold")
for i, v in enumerate(ablation_confs):
    axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

axes[1].barh(ablation_names, ablation_lats, color=ablation_colors, edgecolor="white")
axes[1].set_xlabel("Latency (seconds)", fontsize=11)
axes[1].set_title("Ablation: Latency by Configuration", fontsize=12, fontweight="bold")
for i, v in enumerate(ablation_lats):
    axes[1].text(v + 0.005, i, f"{v:.3f}s", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "ablation_study.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: ablation_study.png")

# ──────────────────────────────────────────────
# Figure 6: Strategy Distribution Pie Chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
strategy_counts = {}
for r in results:
    s = r.get("strategy_used", "unknown")
    strategy_counts[s] = strategy_counts.get(s, 0) + 1

labels = list(strategy_counts.keys())
sizes = list(strategy_counts.values())
pie_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"][:len(labels)]

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=pie_colors, startangle=90, textprops={"fontsize": 11})
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title("AARAG: Strategy Distribution Across All Queries", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "strategy_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: strategy_distribution.png")

# ──────────────────────────────────────────────
# Figure 7: Confidence vs Latency Scatter
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for r in results:
    c = r.get("expected_complexity", "unknown")
    color_map = {"no_retrieval": "#3498db", "single_step": "#2ecc71", "multi_step": "#e67e22", "graph_global": "#9b59b6"}
    ax.scatter(r["latency"], r["confidence"], c=color_map.get(c, "#95a5a6"), s=100, alpha=0.7, edgecolors="white", linewidth=0.5)

ax.set_xlabel("Latency (seconds)", fontsize=11)
ax.set_ylabel("Confidence", fontsize=11)
ax.set_title("AARAG: Confidence vs Latency by Complexity Level", fontsize=13, fontweight="bold")
legend_elements = [
    plt.scatter([], [], c="#3498db", s=80, label="No Retrieval"),
    plt.scatter([], [], c="#2ecc71", s=80, label="Single-Step"),
    plt.scatter([], [], c="#e67e22", s=80, label="Multi-Step"),
    plt.scatter([], [], c="#9b59b6", s=80, label="Graph-Global"),
]
ax.legend(handles=legend_elements, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "confidence_vs_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: confidence_vs_latency.png")

print(f"\nAll 7 figures saved to {FIG_DIR}/")
