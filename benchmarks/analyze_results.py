#!/usr/bin/env python3
"""Analyze τ³-bench simulation results: baseline (llm_agent) vs guarded (harness_agent).

Reads the τ³-bench results.json files and produces a comparison table.

Usage:
    python benchmarks/analyze_results.py \\
        --baseline path/to/baseline/results.json \\
        --guarded  path/to/guarded/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    """Load a τ³-bench results.json file."""
    with open(path) as f:
        return json.load(f)


def extract_task_metrics(results: dict) -> list[dict]:
    """Extract per-task metrics from a τ³-bench results.json."""
    tasks = []
    simulations = results.get("simulations", [])

    for sim in simulations:
        task_id = sim.get("task_id", "unknown")
        agent_cost = sim.get("agent_cost") or 0.0
        user_cost = sim.get("user_cost") or 0.0

        # Extract reward from reward_info
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward", 0.0)

        # Extract token usage from messages
        messages = sim.get("messages", [])
        total_tokens = 0
        turn_count = 0

        for msg in messages:
            usage = msg.get("usage")
            if usage:
                total_tokens += usage.get("completion_tokens", 0) or 0
                total_tokens += usage.get("prompt_tokens", 0) or 0
            if msg.get("role") == "assistant":
                turn_count += 1

        # Check termination reason
        term_reason = sim.get("termination_reason", "")
        is_infra_error = "INFRA" in str(term_reason).upper() if term_reason else False

        tasks.append({
            "task_id": str(task_id),
            "total_tokens": total_tokens,
            "agent_cost": agent_cost,
            "user_cost": user_cost,
            "turn_count": turn_count,
            "reward": reward,
            "passed": reward >= 1.0,
            "is_infra_error": is_infra_error,
            "duration": sim.get("duration", 0),
        })

    return tasks


def print_comparison(
    baseline_tasks: list[dict],
    guarded_tasks: list[dict],
    baseline_path: Path,
    guarded_path: Path,
) -> None:
    """Print a formatted comparison table."""

    def summarize(tasks: list[dict]) -> dict:
        if not tasks:
            return {}
        evaluated = [t for t in tasks if not t["is_infra_error"]]
        return {
            "total": len(tasks),
            "evaluated": len(evaluated),
            "infra_errors": len(tasks) - len(evaluated),
            "tokens": sum(t["total_tokens"] for t in tasks),
            "agent_cost": sum(t["agent_cost"] for t in tasks),
            "user_cost": sum(t["user_cost"] for t in tasks),
            "turns": sum(t["turn_count"] for t in tasks),
            "pass_rate": (
                sum(1 for t in evaluated if t["passed"]) / len(evaluated)
                if evaluated else 0
            ),
            "avg_tokens": (
                statistics.mean(t["total_tokens"] for t in tasks)
                if tasks else 0
            ),
            "max_tokens": max((t["total_tokens"] for t in tasks), default=0),
            "avg_cost": (
                statistics.mean(t["agent_cost"] for t in tasks)
                if tasks else 0
            ),
        }

    b = summarize(baseline_tasks)
    g = summarize(guarded_tasks)

    token_diff = (
        (1 - g["tokens"] / b["tokens"]) * 100 if b["tokens"] > 0 else 0
    )
    cost_diff = (
        (1 - g["agent_cost"] / b["agent_cost"]) * 100
        if b["agent_cost"] > 0 else 0
    )

    print()
    print("═" * 72)
    print("  τ³-bench Results: Baseline (llm_agent) vs Guarded (harness_agent)")
    print("═" * 72)
    print(f"  Baseline: {baseline_path.name}")
    print(f"  Guarded:  {guarded_path.name}")
    print()
    print(f"  {'Metric':<30} {'Baseline':>15} {'Guarded':>15} {'Δ':>8}")
    print("  " + "─" * 68)
    print(f"  {'Tasks Total':<30} {b['total']:>15} {g['total']:>15}")
    print(f"  {'Tasks Evaluated':<30} {b['evaluated']:>15} {g['evaluated']:>15}")
    print(f"  {'Infra Errors (trips)':<30} {b['infra_errors']:>15} {g['infra_errors']:>15}")
    print(f"  {'Pass^1 (evaluated)':<30} {b['pass_rate']:>14.1%} {g['pass_rate']:>14.1%}")
    print("  " + "─" * 68)
    print(f"  {'Total Agent Cost ($)':<30} {b['agent_cost']:>14.4f} {g['agent_cost']:>14.4f} {cost_diff:>7.1f}%")
    print(f"  {'Avg Cost/Task ($)':<30} {b['avg_cost']:>14.4f} {g['avg_cost']:>14.4f}")
    print(f"  {'Total Tokens':<30} {b['tokens']:>15,} {g['tokens']:>15,} {token_diff:>7.1f}%")
    print(f"  {'Avg Tokens/Task':<30} {b['avg_tokens']:>15,.0f} {g['avg_tokens']:>15,.0f}")
    print(f"  {'Max Tokens/Task':<30} {b['max_tokens']:>15,} {g['max_tokens']:>15,}")
    print(f"  {'Total Turns':<30} {b['turns']:>15,} {g['turns']:>15,}")
    print("  " + "─" * 68)

    # Summary
    print()
    if cost_diff > 0:
        savings = b["agent_cost"] - g["agent_cost"]
        print(f"  💰 State-harness saved {cost_diff:.1f}% of agent cost (${savings:.4f})")
    if token_diff > 0:
        print(f"  📉 State-harness reduced total tokens by {token_diff:.1f}%")
    if g["infra_errors"] > 0:
        print(f"  🛡️  Circuit breaker tripped on {g['infra_errors']}/{g['total']} tasks")
        print(f"     These were tasks where token energy kept escalating — runaway prevention in action.")

    # Per-task cost chart
    print()
    print("  Per-Task Cost ($) — Top 10 Most Expensive")
    print("  " + "─" * 60)

    all_tasks = (
        [(t, "B") for t in baseline_tasks]
        + [(t, "G") for t in guarded_tasks]
    )
    all_tasks.sort(key=lambda x: x[0]["agent_cost"], reverse=True)

    max_cost = all_tasks[0][0]["agent_cost"] if all_tasks else 1
    bar_width = 40

    for t, label in all_tasks[:10]:
        bar_len = int(t["agent_cost"] / max_cost * bar_width) if max_cost > 0 else 0
        bar = "█" * bar_len
        tag = "Baseline" if label == "B" else "Guarded "
        status = "✅" if t["passed"] else ("🛡️" if t["is_infra_error"] else "❌")
        print(
            f"  {tag} T{t['task_id']:>3}: {bar:<{bar_width}} "
            f"${t['agent_cost']:.4f} {status}"
        )

    print()
    print("═" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Compare τ³-bench baseline vs guarded results"
    )
    parser.add_argument("--baseline", type=Path, required=True,
                       help="Path to baseline results.json")
    parser.add_argument("--guarded", type=Path, required=True,
                       help="Path to guarded results.json")
    args = parser.parse_args()

    baseline_data = load_results(args.baseline)
    guarded_data = load_results(args.guarded)

    baseline_tasks = extract_task_metrics(baseline_data)
    guarded_tasks = extract_task_metrics(guarded_data)

    print_comparison(baseline_tasks, guarded_tasks, args.baseline, args.guarded)


if __name__ == "__main__":
    main()
