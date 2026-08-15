# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""CLI tool for state-harness.

Provides command-line access to failure analysis, batch reporting,
and guard state inspection.

Usage::

    # Analyze a saved guard state (JSON)
    state-harness analyze report.json

    # Batch analyze multiple reports
    state-harness batch --dir ./reports/ --output results.csv

    # Show version
    state-harness --version
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state-harness",
        description="Runtime safety net for LLM agents. Analyze failure reports from the command line.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── analyze ──
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a saved failure report (JSON).",
    )
    analyze.add_argument(
        "input",
        type=str,
        help="Path to a JSON file containing a FailureReport dict (from report.to_json()).",
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output as JSON instead of human-readable format.",
    )
    analyze.add_argument(
        "--otel",
        action="store_true",
        dest="output_otel",
        help="Output as OpenTelemetry span attributes.",
    )

    # ── batch ──
    batch = subparsers.add_parser(
        "batch",
        help="Batch analyze multiple report files and output CSV.",
    )
    batch.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing JSON report files.",
    )
    batch.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: stdout).",
    )
    batch.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for matching report files (default: *.json).",
    )

    # ── simulate ──
    simulate = subparsers.add_parser(
        "simulate",
        help="Simulate a token trajectory and show what the guard would do.",
    )
    simulate.add_argument(
        "tokens",
        type=int,
        nargs="+",
        help="Space-separated token counts per turn (e.g., 1000 1500 2000 3500 8000).",
    )
    simulate.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Token budget (default: 100000).",
    )
    simulate.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Growth ratio threshold (default: 2.0).",
    )
    simulate.add_argument(
        "--window",
        type=int,
        default=3,
        help="Consecutive escalating turns before trip (default: 3).",
    )

    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a single report file."""
    from state_harness.diagnostics import FailureReport, FailurePattern, Suggestion

    path = Path(args.input)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1

    # Reconstruct a FailureReport from the dict
    report = FailureReport(
        pattern=FailurePattern(data.get("pattern", "unknown")),
        confidence=data.get("confidence", 0.0),
        total_tokens=data.get("total_tokens", 0),
        total_steps=data.get("total_steps", 0),
        is_tripped=data.get("is_tripped", False),
        is_frozen=data.get("is_frozen", False),
        baseline_tokens=data.get("baseline_tokens"),
        peak_ratio=data.get("peak_ratio"),
        energy_trajectory=data.get("energy_trajectory", []),
        drift_trajectory=data.get("drift_trajectory", []),
        cost_estimate_usd=data.get("cost_estimate_usd"),
        projected_cost_usd=data.get("projected_cost_usd"),
        suggestions=[
            Suggestion(s["priority"], s["action"], s["rationale"])
            for s in data.get("suggestions", [])
        ],
        evidence=data.get("evidence", []),
    )

    if args.output_json:
        print(report.to_json())
    elif args.output_otel:
        attrs = report.to_otel_attributes()
        for k, v in sorted(attrs.items()):
            print(f"{k}: {v}")
    else:
        print(report)

    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    """Batch analyze multiple report files."""
    from state_harness.diagnostics import FailureReport, FailurePattern, Suggestion

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"Error: Not a directory: {dir_path}", file=sys.stderr)
        return 1

    files = sorted(glob.glob(str(dir_path / args.pattern)))
    if not files:
        print(f"No files matching '{args.pattern}' in {dir_path}", file=sys.stderr)
        return 1

    output = sys.stdout
    if args.output:
        output = open(args.output, "w")

    try:
        output.write("file," + FailureReport.csv_header() + "\n")
        for filepath in files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                report = FailureReport(
                    pattern=FailurePattern(data.get("pattern", "unknown")),
                    confidence=data.get("confidence", 0.0),
                    total_tokens=data.get("total_tokens", 0),
                    total_steps=data.get("total_steps", 0),
                    is_tripped=data.get("is_tripped", False),
                    is_frozen=data.get("is_frozen", False),
                    baseline_tokens=data.get("baseline_tokens"),
                    peak_ratio=data.get("peak_ratio"),
                    cost_estimate_usd=data.get("cost_estimate_usd"),
                    projected_cost_usd=data.get("projected_cost_usd"),
                    suggestions=[
                        Suggestion(s["priority"], s["action"], s["rationale"])
                        for s in data.get("suggestions", [])
                    ],
                    evidence=data.get("evidence", []),
                )
                basename = os.path.basename(filepath)
                output.write(f"{basename},{report.to_csv_row()}\n")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping {filepath}: {e}", file=sys.stderr)
    finally:
        if args.output and output is not sys.stdout:
            output.close()

    if args.output:
        print(f"Wrote {len(files)} reports to {args.output}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Simulate a token trajectory."""
    from state_harness import GrowthRatioGuard, StabilityViolation, BudgetExhausted
    from state_harness.diagnostics import FailureReport

    guard = GrowthRatioGuard(
        token_budget=args.budget,
        ratio_threshold=args.threshold,
        window=args.window,
    )

    print(f"Simulating {len(args.tokens)} turns with budget={args.budget:,}, "
          f"threshold={args.threshold}, window={args.window}")
    print(f"Tokens: {args.tokens}")
    print()

    with guard:
        for i, tokens in enumerate(args.tokens, 1):
            try:
                status = guard.record_step(tokens_used=tokens)
                ratio = guard.current_ratio or 0
                print(f"  Turn {i:3d}: {tokens:>8,} tokens  │  ratio: {ratio:.2f}×  │  {status}")
            except StabilityViolation:
                print(f"  Turn {i:3d}: {tokens:>8,} tokens  │  ⚠️ STABILITY TRIPPED")
                break
            except BudgetExhausted:
                print(f"  Turn {i:3d}: {tokens:>8,} tokens  │  💸 BUDGET EXHAUSTED")
                break

    print()
    report = FailureReport.from_guard(guard)
    print(report)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        from state_harness import __version__
        print(f"state-harness {__version__}")
        return 0

    if args.command == "analyze":
        return _cmd_analyze(args)
    elif args.command == "batch":
        return _cmd_batch(args)
    elif args.command == "simulate":
        return _cmd_simulate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
