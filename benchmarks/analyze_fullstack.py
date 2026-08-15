"""Full-stack benchmark analysis — compares 4 experimental conditions.

Usage:
    python3 analyze_fullstack.py /path/to/results_dir

Reads results from all 4 conditions and generates:
1. Cross-condition comparison table (pass rate, token savings, precision)
2. Per-condition breakdown
3. Specific SWE-bench FP tracking (if applicable)
"""

import json
import os
import sys
from typing import Dict, List, Any
from collections import defaultdict


def load_tau3_results(results_dir: str) -> Dict[str, Any]:
    """Load τ³-bench results from a condition directory."""
    rfile = os.path.join(results_dir, "results.json")
    if not os.path.exists(rfile):
        return {"error": f"No results.json in {results_dir}"}

    with open(rfile) as f:
        data = json.load(f)

    sims = data.get("simulations", [])
    rewards = [(s.get("reward_info") or {}).get("reward", 0) for s in sims]
    costs = [s.get("agent_cost", 0) or 0 for s in sims]
    tokens = [s.get("agent_tokens", 0) or 0 for s in sims]
    infra = sum(1 for s in sims if s.get("termination_reason") == "infrastructure_error")
    harness_trips = sum(1 for s in sims if "harness" in str(s.get("termination_reason", "")))

    pass_count = sum(1 for r in rewards if r > 0)
    total = len(sims)

    return {
        "total_runs": total,
        "pass_count": pass_count,
        "pass_rate": pass_count / max(total, 1) * 100,
        "avg_cost": sum(costs) / max(len(costs), 1),
        "total_cost": sum(costs),
        "avg_tokens": sum(tokens) / max(len(tokens), 1),
        "total_tokens": sum(tokens),
        "infra_errors": infra,
        "harness_trips": harness_trips,
        "rewards": rewards,
        "costs": costs,
        "tokens": tokens,
    }


def analyze_tau3(results_dir: str):
    """Analyze τ³-bench results across all 4 conditions."""
    conditions = {}

    for name in sorted(os.listdir(results_dir)):
        cond_dir = os.path.join(results_dir, name)
        if not os.path.isdir(cond_dir):
            continue
        rfile = os.path.join(cond_dir, "results.json")
        if not os.path.exists(rfile):
            continue
        conditions[name] = load_tau3_results(cond_dir)

    if not conditions:
        print(f"No results found in {results_dir}")
        return

    # Print comparison table
    print("\n" + "=" * 80)
    print("  FULL-STACK BENCHMARK RESULTS — τ³-bench Airline")
    print("=" * 80)

    header = f"  {'Condition':<40s} {'Runs':>5s}  {'Pass%':>6s}  {'AvgCost':>9s}  {'AvgTokens':>10s}  {'Trips':>5s}"
    print(header)
    print(f"  {'─' * 78}")

    baseline_tokens = None
    for name, r in conditions.items():
        if "baseline" in name.lower():
            baseline_tokens = r["avg_tokens"]
            break

    for name, r in conditions.items():
        savings = ""
        if baseline_tokens and baseline_tokens > 0 and "baseline" not in name.lower():
            saved = (1 - r["avg_tokens"] / baseline_tokens) * 100
            savings = f" ({saved:+.1f}%)"

        print(
            f"  {name:<40s} "
            f"{r['total_runs']:>5d}  "
            f"{r['pass_rate']:>5.1f}%  "
            f"${r['avg_cost']:>8.4f}  "
            f"{r['avg_tokens']:>10.0f}{savings}  "
            f"{r['harness_trips']:>5d}"
        )

    print(f"  {'─' * 78}")
    total_cost = sum(r["total_cost"] for r in conditions.values())
    print(f"  Total API cost: ${total_cost:.2f}")
    print()

    # Precision analysis (for harness conditions)
    print("  PRECISION ANALYSIS (harness conditions only)")
    print(f"  {'─' * 60}")
    for name, r in conditions.items():
        if r["harness_trips"] == 0:
            continue
        # True positives: trips where baseline would have failed
        # (approximate: rewards=0 AND tripped)
        trips = r["harness_trips"]
        print(f"  {name}: {trips} trips")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_fullstack.py /path/to/results_dir")
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    analyze_tau3(results_dir)


if __name__ == "__main__":
    main()
