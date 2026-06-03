#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Analyze local model benchmark results.

Reads JSON results from benchmark runs and produces:
  1. Per-model summary tables
  2. Cross-model comparison
  3. Harness value metrics (tokens saved, false positive rate, detection latency)
  4. LaTeX-ready tables for the paper
  5. Frontier vs local comparison (if frontier data is available)

Usage:
    python benchmarks/local_models/analyze_local_results.py
    python benchmarks/local_models/analyze_local_results.py --latex
    python benchmarks/local_models/analyze_local_results.py --results-dir benchmark_results/local_models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all result files, grouped by model."""
    model_results: dict[str, list[dict]] = {}

    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        model = data["metadata"]["model"]
        if model not in model_results:
            model_results[model] = []

        model_results[model].extend(data["results"])

    return model_results


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from a list of run results."""
    conditions = ["baseline", "naive_cap", "harness"]
    metrics: dict[str, Any] = {}

    for cond in conditions:
        cond_runs = [r for r in results if r["condition"] == cond]
        if not cond_runs:
            continue

        n = len(cond_runs)
        successes = sum(1 for r in cond_runs if r["success"])
        total_tokens = sum(r["total_tokens"] for r in cond_runs)
        total_time = sum(r["wall_time_seconds"] for r in cond_runs)
        trips = sum(1 for r in cond_runs if r.get("harness_tripped", False))

        # Per-difficulty breakdown
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            diff_runs = [r for r in cond_runs if r["difficulty"] == diff]
            if diff_runs:
                by_diff[diff] = {
                    "n": len(diff_runs),
                    "successes": sum(1 for r in diff_runs if r["success"]),
                    "total_tokens": sum(r["total_tokens"] for r in diff_runs),
                    "avg_tokens": sum(r["total_tokens"] for r in diff_runs) / len(diff_runs),
                    "avg_turns": sum(r["turns_used"] for r in diff_runs) / len(diff_runs),
                    "total_time": sum(r["wall_time_seconds"] for r in diff_runs),
                }

        metrics[cond] = {
            "n": n,
            "successes": successes,
            "pass_rate": 100 * successes / n if n > 0 else 0,
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / n if n > 0 else 0,
            "total_time": total_time,
            "avg_time": total_time / n if n > 0 else 0,
            "trips": trips,
            "by_difficulty": by_diff,
        }

    # Harness value metrics
    baseline_by_task = {r["task_name"]: r for r in results if r["condition"] == "baseline"}
    harness_by_task = {r["task_name"]: r for r in results if r["condition"] == "harness"}

    tokens_saved = 0
    time_saved = 0
    false_positives = 0
    true_positives = 0
    detection_latencies = []  # turns before trip on failed tasks

    for task_name in baseline_by_task:
        b = baseline_by_task[task_name]
        h = harness_by_task.get(task_name)
        if not h:
            continue

        if h.get("harness_tripped", False):
            tokens_saved += max(0, b["total_tokens"] - h.get("tokens_at_trip", h["total_tokens"]))
            time_saved += max(0, b["wall_time_seconds"] - h["wall_time_seconds"])
            if b["success"]:
                false_positives += 1
            else:
                true_positives += 1
                if h.get("trip_turn"):
                    detection_latencies.append(h["trip_turn"])

    baseline_total_tokens = sum(r["total_tokens"] for r in baseline_by_task.values())
    baseline_total_time = sum(r["wall_time_seconds"] for r in baseline_by_task.values())

    metrics["harness_value"] = {
        "tokens_saved": tokens_saved,
        "tokens_saved_pct": 100 * tokens_saved / baseline_total_tokens if baseline_total_tokens > 0 else 0,
        "time_saved": time_saved,
        "time_saved_pct": 100 * time_saved / baseline_total_time if baseline_total_time > 0 else 0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
        "avg_detection_latency": sum(detection_latencies) / len(detection_latencies) if detection_latencies else 0,
    }

    return metrics


def print_model_summary(model: str, metrics: dict):
    """Print a human-readable summary for one model."""
    print(f"\n{'═' * 70}")
    print(f"  {model}")
    print(f"{'═' * 70}")

    for cond in ["baseline", "naive_cap", "harness"]:
        m = metrics.get(cond, {})
        if not m:
            continue
        print(f"\n  {cond.upper():<12s}  Pass: {m['successes']}/{m['n']} ({m['pass_rate']:.1f}%)  "
              f"Tokens: {m['avg_tokens']:,.0f} avg  Time: {m['avg_time']:.1f}s avg  "
              f"Trips: {m.get('trips', 0)}")

        for diff in ["easy", "medium", "hard"]:
            d = m.get("by_difficulty", {}).get(diff, {})
            if d:
                pr = 100 * d["successes"] / d["n"] if d["n"] > 0 else 0
                print(f"    {diff:<8s}  {d['successes']}/{d['n']} ({pr:.0f}%)  "
                      f"{d['avg_tokens']:,.0f} avg tok  {d['avg_turns']:.1f} avg turns")

    hv = metrics.get("harness_value", {})
    if hv:
        print(f"\n  HARNESS VALUE:")
        print(f"    Tokens saved:   {hv['tokens_saved']:>10,d} ({hv['tokens_saved_pct']:.1f}%)")
        print(f"    Time saved:     {hv['time_saved']:>10.1f}s ({hv['time_saved_pct']:.1f}%)")
        print(f"    True positives: {hv['true_positives']}")
        print(f"    False positives:{hv['false_positives']}")
        print(f"    Precision:      {hv['precision']:.1%}")
        print(f"    Avg detection:  turn {hv['avg_detection_latency']:.1f}")


def print_cross_model_comparison(all_metrics: dict[str, dict]):
    """Print a cross-model comparison table."""
    print(f"\n\n{'═' * 80}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'═' * 80}\n")

    models = sorted(all_metrics.keys())

    # Header
    print(f"  {'Model':<20s} {'Cond':<10s} {'Pass%':<8s} {'Avg Tok':<10s} "
          f"{'Avg Time':<10s} {'Trips':<6s}")
    print(f"  {'─' * 70}")

    for model in models:
        m = all_metrics[model]
        for cond in ["baseline", "naive_cap", "harness"]:
            cm = m.get(cond, {})
            if not cm:
                continue
            model_label = model if cond == "baseline" else ""
            print(f"  {model_label:<20s} {cond:<10s} {cm['pass_rate']:<8.1f} "
                  f"{cm['avg_tokens']:<10,.0f} {cm['avg_time']:<10.1f} "
                  f"{cm.get('trips', 0):<6d}")
        print()

    # Harness value comparison
    print(f"\n  {'Model':<20s} {'Tok Saved%':<12s} {'Time Saved%':<12s} "
          f"{'TP':<5s} {'FP':<5s} {'Precision':<10s}")
    print(f"  {'─' * 70}")

    for model in models:
        hv = all_metrics[model].get("harness_value", {})
        if hv:
            print(f"  {model:<20s} {hv['tokens_saved_pct']:<12.1f} "
                  f"{hv['time_saved_pct']:<12.1f} {hv['true_positives']:<5d} "
                  f"{hv['false_positives']:<5d} {hv['precision']:<10.1%}")


def generate_latex_tables(all_metrics: dict[str, dict]) -> str:
    """Generate LaTeX tables for the paper."""
    models = sorted(all_metrics.keys())

    latex = []
    latex.append("% ── Auto-generated by analyze_local_results.py ──")
    latex.append("")

    # Table 1: Cross-model comparison
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Local model benchmark results across 20 multi-turn coding tasks.}")
    latex.append(r"\label{tab:local-models}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{ll rrr r}")
    latex.append(r"\toprule")
    latex.append(r"Model & Condition & Pass\% & Avg Tok & Avg Time & Trips \\")
    latex.append(r"\midrule")

    for i, model in enumerate(models):
        m = all_metrics[model]
        # Short model name
        short = model.split(":")[0] if ":" in model else model
        for j, cond in enumerate(["baseline", "naive\_cap", "harness"]):
            cond_key = cond.replace(r"\_", "_")
            cm = m.get(cond_key, {})
            if not cm:
                continue
            model_label = short if j == 0 else ""
            latex.append(
                f"{model_label} & {cond} & {cm['pass_rate']:.1f} & "
                f"{cm['avg_tokens']:,.0f} & {cm['avg_time']:.1f}s & "
                f"{cm.get('trips', 0)} \\\\"
            )
        if i < len(models) - 1:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")

    # Table 2: Harness value
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Harness value metrics on local models.}")
    latex.append(r"\label{tab:local-harness-value}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l rr rr r}")
    latex.append(r"\toprule")
    latex.append(r"Model & Tok Saved & \% & TP & FP & Precision \\")
    latex.append(r"\midrule")

    for model in models:
        short = model.split(":")[0] if ":" in model else model
        hv = all_metrics[model].get("harness_value", {})
        if hv:
            latex.append(
                f"{short} & {hv['tokens_saved']:,d} & {hv['tokens_saved_pct']:.1f}\\% & "
                f"{hv['true_positives']} & {hv['false_positives']} & "
                f"{hv['precision']:.1%} \\\\"
            )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Analyze local model benchmark results")
    parser.add_argument(
        "--results-dir", type=str,
        default=str(Path(__file__).parent.parent.parent / "benchmark_results" / "local_models"),
        help="Directory containing JSON result files",
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    parser.add_argument("--json", action="store_true", help="Output aggregated JSON")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1

    model_results = load_results(results_dir)

    if not model_results:
        print(f"❌ No result files found in {results_dir}")
        return 1

    print(f"Found results for {len(model_results)} model(s): {', '.join(model_results.keys())}")

    all_metrics = {}
    for model, results in model_results.items():
        metrics = compute_metrics(results)
        all_metrics[model] = metrics
        print_model_summary(model, metrics)

    if len(model_results) > 1:
        print_cross_model_comparison(all_metrics)

    if args.latex:
        print(f"\n\n{'═' * 70}")
        print(f"  LaTeX TABLES")
        print(f"{'═' * 70}\n")
        print(generate_latex_tables(all_metrics))

    if args.json:
        output_file = results_dir / "aggregated_metrics.json"
        with open(output_file, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nAggregated metrics saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
