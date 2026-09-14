#!/usr/bin/env python3
"""AARAG Full Pipeline Runner — Runs the complete pipeline with detailed logging.

Usage:
    python run_pipeline.py                    # Run with defaults
    python run_pipeline.py --data data/my_docs/   # Custom data path
    python run_pipeline.py --query "What is ML?"   # Single query
    python run_pipeline.py --eval              # Run evaluation
    python run_pipeline.py --demo              # Launch demo
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

from src.aarag import AARAG
from src.utils.logger import get_logger

console = Console()
logger = get_logger("aarag.runner", log_file="results/logs/pipeline.log")


def print_banner():
    """Print the AARAG banner."""
    banner = """
[bold blue]
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗  █████╗ ██████╗  ██████╗                          ║
    ║    ██╔══██╗██╔══██╗██╔══██╗██╔════╝                          ║
    ║    ███████║███████║██████╔╝██║  ███╗                         ║
    ║    ██╔══██║██╔══██║██╔══██╗██║   ██║                         ║
    ║    ██║  ██║██║  ██║██║  ██║╚██████╔╝                         ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                        ║
    ║                                                               ║
    ║     Adaptive Agentic Retrieval-Augmented Generation           ║
    ║     M.Research AI Capstone Project 2026                       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
[/bold blue]
"""
    console.print(banner)


def run_ingestion(aarag: AARAG, data_path: str):
    """Run document ingestion with progress tracking."""
    console.print("\n[bold cyan]═══ PHASE 1: Document Ingestion ═══[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=None)

        stats = aarag.ingest(data_path)

        progress.update(task, completed=True)

    # Display results
    table = Table(title="Ingestion Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)
    return stats


def run_queries(aarag: AARAG, queries: list):
    """Run multiple queries with detailed output."""
    console.print("\n[bold cyan]═══ PHASE 2: Query Processing ═══[/bold cyan]\n")

    results = []

    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold yellow]Query {i}/{len(queries)}:[/bold yellow] {query}")
        console.print("─" * 60)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing through 5 layers...", total=None)
            response = aarag.query(query)
            progress.update(task, completed=True)

        # Display results
        console.print(Panel(
            f"[bold]{response.answer}[/bold]",
            title="💡 Answer",
            border_style="green",
        ))

        # Metrics table
        metrics_table = Table(box=box.SIMPLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics_table.add_row("Confidence", f"{response.confidence:.2f}")
        metrics_table.add_row("Latency", f"{response.latency:.3f}s")
        metrics_table.add_row("Strategy", response.strategy_used)
        metrics_table.add_row("Sources", str(len(response.sources)))

        console.print(metrics_table)

        # Layer details
        layer_table = Table(title="Layer Details", box=box.SIMPLE)
        layer_table.add_column("Layer", style="cyan")
        layer_table.add_column("Decision", style="yellow")
        layer_table.add_column("Confidence", style="green")

        routing = response.routing_decision
        level_names = {0: "No Retrieval", 1: "Single-Step", 2: "Multi-Step", 3: "Graph-Global"}
        layer_table.add_row(
            "L2: Router",
            level_names.get(routing.get("level", 0), "Unknown"),
            f"{routing.get('confidence', 0):.2f}",
        )

        corrective = response.corrective_result
        layer_table.add_row(
            "L3: CRAG",
            corrective.get("action", "N/A"),
            f"{corrective.get('confidence', 0):.2f}",
        )

        if response.reflection_result:
            ref = response.reflection_result
            layer_table.add_row("L4: Reflection", ref.get("relevance", "N/A"), f"{ref.get('final_confidence', 0):.2f}")

        agent = response.agent_response
        layer_table.add_row(
            "L5: Agent",
            f"{agent.get('total_steps', 0)} steps",
            f"{agent.get('confidence', 0):.2f}",
        )

        console.print(layer_table)
        results.append(response)

    return results


def run_evaluation(aarag: AARAG, max_samples: int = 100):
    """Run evaluation with progress tracking."""
    console.print("\n[bold cyan]═══ PHASE 3: Evaluation ═══[/bold cyan]\n")

    from src.evaluation.evaluate import run_full_evaluation

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running evaluation...", total=None)

        results = run_full_evaluation(
            aarag=aarag,
            eval_dir="data/processed",
            output_dir="results",
            max_samples=max_samples,
            run_ablation=True,
        )

        progress.update(task, completed=True)

    # Display summary
    if results:
        table = Table(title="Evaluation Results", box=box.ROUNDED)
        table.add_column("System", style="cyan")
        table.add_column("EM", style="green")
        table.add_column("F1", style="green")
        table.add_column("Faithfulness", style="green")
        table.add_column("Latency", style="yellow")

        for dataset_name, dataset_results in results.items():
            if dataset_name == "ablation":
                continue
            for system_name, metrics in dataset_results.items():
                if isinstance(metrics, dict):
                    table.add_row(
                        f"{system_name}",
                        f"{metrics.get('exact_match', 0):.3f}",
                        f"{metrics.get('f1', 0):.3f}",
                        f"{metrics.get('faithfulness', 0):.3f}",
                        f"{metrics.get('avg_latency', 0):.2f}s",
                    )

        console.print(table)

    return results


def main():
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(description="AARAG Full Pipeline Runner")
    parser.add_argument("--data", type=str, help="Path to data for ingestion")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--queries", type=str, nargs="+", help="Multiple queries to process")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--eval-samples", type=int, default=100, help="Max evaluation samples")
    parser.add_argument("--demo", action="store_true", help="Launch Streamlit demo")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file path")
    parser.add_argument("--save", type=str, help="Save path for AARAG model")
    parser.add_argument("--load", type=str, help="Load path for AARAG model")

    args = parser.parse_args()

    # Ensure log directory exists
    Path("results/logs").mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # Initialize AARAG
    # ──────────────────────────────────────────────
    console.print("\n[bold cyan]═══ INITIALIZATION ═══[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing AARAG system...", total=None)
        aarag = AARAG(config_path=args.config)
        progress.update(task, completed=True)

    # Load pre-trained models if specified
    if args.load:
        console.print(f"\nLoading pre-trained models from: {args.load}")
        aarag.load(args.load)

    # ──────────────────────────────────────────────
    # Run ingestion if data provided
    # ──────────────────────────────────────────────
    if args.data:
        run_ingestion(aarag, args.data)

    # ──────────────────────────────────────────────
    # Run queries
    # ──────────────────────────────────────────────
    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries:
        queries = args.queries
    else:
        # Default example queries
        queries = [
            "What is machine learning?",
            "Compare Python and Java programming languages.",
            "Who is Albert Einstein and what is the theory of relativity?",
        ]
        console.print("[dim]Using default example queries (use --query or --queries to specify)[/dim]")

    if queries:
        results = run_queries(aarag, queries)

    # ──────────────────────────────────────────────
    # Run evaluation
    # ──────────────────────────────────────────────
    if args.eval:
        eval_results = run_evaluation(aarag, args.eval_samples)

    # ──────────────────────────────────────────────
    # Save model
    # ──────────────────────────────────────────────
    if args.save:
        console.print(f"\nSaving AARAG to: {args.save}")
        aarag.save(args.save)

    # ──────────────────────────────────────────────
    # Launch demo
    # ──────────────────────────────────────────────
    if args.demo:
        console.print("\n[bold cyan]═══ LAUNCHING DEMO ═══[/bold cyan]")
        console.print("Opening Streamlit demo at http://localhost:8501")
        os.system("streamlit run demo/app.py --server.port 8501")

    # ──────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────
    console.print("\n[bold green]═══ PIPELINE COMPLETE ═══[/bold green]")
    console.print(f"Logs saved to: results/logs/")
    console.print(f"Results saved to: results/")


if __name__ == "__main__":
    main()
