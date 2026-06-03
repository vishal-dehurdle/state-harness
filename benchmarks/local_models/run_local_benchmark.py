#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Local/edge model benchmark runner for state-harness.

Runs multi-turn coding tasks through Ollama-hosted local models with
three experimental conditions:
  A. Baseline   — Agent runs to completion (max_turns cap)
  B. Naive Cap  — Hard cap at max_turns/2
  C. Harness    — state-harness GrowthRatioGuard monitors + kills spirals

Measures: success rate, total tokens, wall time, wasted compute,
detection latency, and false positive rate.

Usage:
    # Run a specific model
    python benchmarks/local_models/run_local_benchmark.py --model qwen3:4b

    # Dry run (validate logic without full execution)
    python benchmarks/local_models/run_local_benchmark.py --model qwen3:4b --dry-run

    # Run specific difficulty tier
    python benchmarks/local_models/run_local_benchmark.py --model qwen3:4b --difficulty medium

    # Run a single task
    python benchmarks/local_models/run_local_benchmark.py --model qwen3:4b --task fibonacci
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# Add python source to path for state_harness imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from tasks import ALL_TASKS, TASKS_BY_DIFFICULTY, TASKS_BY_NAME, Task

from state_harness import GrowthRatioGuard, StabilityViolation, BudgetExhausted
from state_harness.diagnostics import FailureReport


# ─── Ollama Client ──────────────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class OllamaTurn:
    """Result from a single Ollama turn."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    content: str
    duration_ms: float
    eval_count: int = 0          # tokens generated
    eval_duration_ns: int = 0    # time spent generating
    prompt_eval_count: int = 0   # tokens in prompt
    prompt_eval_duration_ns: int = 0


def call_ollama(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> OllamaTurn:
    """Call Ollama's chat API and return structured result."""
    import urllib.request
    import urllib.error

    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE}. "
            f"Is Ollama running? Error: {e}"
        )

    duration_ms = (time.monotonic() - start) * 1000

    content = data.get("message", {}).get("content", "")

    # Ollama returns token counts in various places
    eval_count = data.get("eval_count", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    prompt_eval_duration = data.get("prompt_eval_duration", 0)

    # Estimate total tokens
    prompt_tokens = prompt_eval_count if prompt_eval_count > 0 else _estimate_tokens(messages)
    completion_tokens = eval_count if eval_count > 0 else _estimate_tokens_str(content)
    total_tokens = prompt_tokens + completion_tokens

    return OllamaTurn(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        content=content,
        duration_ms=duration_ms,
        eval_count=eval_count,
        eval_duration_ns=eval_duration,
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration_ns=prompt_eval_duration,
    )


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate from message text (4 chars ≈ 1 token)."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, total_chars // 4)


def _estimate_tokens_str(text: str) -> int:
    """Rough token estimate from text."""
    return max(1, len(text) // 4)


# ─── Run Conditions ─────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Result from running one task under one condition."""
    task_name: str
    model: str
    condition: str  # "baseline", "naive_cap", "harness"
    difficulty: str
    success: bool = False
    turns_used: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_seconds: float = 0.0
    harness_tripped: bool = False
    trip_turn: Optional[int] = None
    failure_pattern: Optional[str] = None
    tokens_at_trip: int = 0  # tokens used when harness would have tripped
    error: Optional[str] = None
    turn_tokens: list[int] = field(default_factory=list)
    growth_ratios: list[float] = field(default_factory=list)
    eval_tokens_per_sec: float = 0.0  # generation speed


def run_task_baseline(task: Task, model: str, max_turns: int = 20) -> RunResult:
    """Condition A: Run with no monitoring, just a hard step cap."""
    result = RunResult(
        task_name=task.name,
        model=model,
        condition="baseline",
        difficulty=task.difficulty,
    )

    conversation: list[dict] = []
    start_time = time.monotonic()
    total_eval_count = 0
    total_eval_duration_ns = 0

    for i, turn_msg in enumerate(task.turns):
        if i >= max_turns:
            break

        conversation.append(turn_msg)

        try:
            turn = call_ollama(model, conversation)
            result.turns_used += 1
            result.total_tokens += turn.total_tokens
            result.prompt_tokens += turn.prompt_tokens
            result.completion_tokens += turn.completion_tokens
            result.turn_tokens.append(turn.total_tokens)
            total_eval_count += turn.eval_count
            total_eval_duration_ns += turn.eval_duration_ns

            conversation.append({"role": "assistant", "content": turn.content})

        except Exception as e:
            result.error = f"Turn {i+1}: {type(e).__name__}: {e}"
            break

    result.wall_time_seconds = time.monotonic() - start_time

    # Calculate generation speed
    if total_eval_duration_ns > 0:
        result.eval_tokens_per_sec = total_eval_count / (total_eval_duration_ns / 1e9)

    # Validate final output
    if conversation and conversation[-1]["role"] == "assistant":
        final_output = conversation[-1]["content"]
        if task.validator:
            try:
                result.success = task.validator(final_output)
            except Exception:
                result.success = False

    return result


def run_task_naive_cap(task: Task, model: str, cap_turns: int = 0) -> RunResult:
    """Condition B: Hard cap at half the task's max turns."""
    if cap_turns <= 0:
        cap_turns = max(1, len(task.turns) // 2)

    result = RunResult(
        task_name=task.name,
        model=model,
        condition="naive_cap",
        difficulty=task.difficulty,
    )

    conversation: list[dict] = []
    start_time = time.monotonic()
    total_eval_count = 0
    total_eval_duration_ns = 0

    for i, turn_msg in enumerate(task.turns):
        if i >= cap_turns:
            break

        conversation.append(turn_msg)

        try:
            turn = call_ollama(model, conversation)
            result.turns_used += 1
            result.total_tokens += turn.total_tokens
            result.prompt_tokens += turn.prompt_tokens
            result.completion_tokens += turn.completion_tokens
            result.turn_tokens.append(turn.total_tokens)
            total_eval_count += turn.eval_count
            total_eval_duration_ns += turn.eval_duration_ns

            conversation.append({"role": "assistant", "content": turn.content})

        except Exception as e:
            result.error = f"Turn {i+1}: {type(e).__name__}: {e}"
            break

    result.wall_time_seconds = time.monotonic() - start_time

    if total_eval_duration_ns > 0:
        result.eval_tokens_per_sec = total_eval_count / (total_eval_duration_ns / 1e9)

    if conversation and conversation[-1]["role"] == "assistant":
        final_output = conversation[-1]["content"]
        if task.validator:
            try:
                result.success = task.validator(final_output)
            except Exception:
                result.success = False

    return result


def run_task_harness(task: Task, model: str, max_turns: int = 20) -> RunResult:
    """Condition C: Run with state-harness GrowthRatioGuard monitoring."""
    result = RunResult(
        task_name=task.name,
        model=model,
        condition="harness",
        difficulty=task.difficulty,
    )

    # Configure guard for local models — tighter thresholds than cloud
    # because local models spiral faster and recover less
    guard = GrowthRatioGuard(
        token_budget=100_000,       # Lower budget for local models
        ratio_threshold=1.8,        # Tighter threshold (small models spiral faster)
        window=3,
        warmup_turns=2,             # Shorter warmup (less data needed)
        budget_gate=3_000,          # Lower gate (local inference is slower)
    )

    conversation: list[dict] = []
    start_time = time.monotonic()
    total_eval_count = 0
    total_eval_duration_ns = 0

    with guard:
        for i, turn_msg in enumerate(task.turns):
            if i >= max_turns:
                break

            conversation.append(turn_msg)

            try:
                turn = call_ollama(model, conversation)
                result.turns_used += 1
                result.total_tokens += turn.total_tokens
                result.prompt_tokens += turn.prompt_tokens
                result.completion_tokens += turn.completion_tokens
                result.turn_tokens.append(turn.total_tokens)
                total_eval_count += turn.eval_count
                total_eval_duration_ns += turn.eval_duration_ns

                # Record step in guard
                try:
                    guard.record_step(tokens_used=turn.total_tokens)
                    if guard.current_ratio is not None:
                        result.growth_ratios.append(guard.current_ratio)
                except StabilityViolation:
                    result.harness_tripped = True
                    result.trip_turn = i + 1
                    result.tokens_at_trip = result.total_tokens
                    break
                except BudgetExhausted:
                    result.harness_tripped = True
                    result.trip_turn = i + 1
                    result.tokens_at_trip = result.total_tokens
                    break

                conversation.append({"role": "assistant", "content": turn.content})

            except Exception as e:
                result.error = f"Turn {i+1}: {type(e).__name__}: {e}"
                break

    result.wall_time_seconds = time.monotonic() - start_time

    if total_eval_duration_ns > 0:
        result.eval_tokens_per_sec = total_eval_count / (total_eval_duration_ns / 1e9)

    # Get failure pattern from diagnostics
    try:
        report = FailureReport.from_guard(guard)
        result.failure_pattern = report.pattern.value
    except Exception:
        result.failure_pattern = "unknown"

    # Validate final output
    if conversation and conversation[-1]["role"] == "assistant":
        final_output = conversation[-1]["content"]
        if task.validator:
            try:
                result.success = task.validator(final_output)
            except Exception:
                result.success = False

    return result


# ─── Benchmark Runner ────────────────────────────────────────────────────

def run_benchmark(
    model: str,
    tasks: list[Task],
    dry_run: bool = False,
) -> list[RunResult]:
    """Run all tasks across all 3 conditions for a given model."""

    total = len(tasks) * 3
    results: list[RunResult] = []

    print(f"\n{'═' * 70}")
    print(f"  Local Model Benchmark — {model}")
    print(f"  Tasks: {len(tasks)} | Conditions: 3 | Total runs: {total}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}\n")

    for task_idx, task in enumerate(tasks):
        print(f"\n{'─' * 60}")
        print(f"  Task {task_idx + 1}/{len(tasks)}: {task.name} ({task.difficulty})")
        print(f"  Turns: {len(task.turns)} | Max: {task.max_turns}")
        print(f"{'─' * 60}")

        # ── Condition A: Baseline ──
        print(f"\n  [A] Baseline...", end=" ", flush=True)
        if dry_run:
            r = RunResult(task.name, model, "baseline", task.difficulty)
            print("SKIPPED (dry run)")
        else:
            r = run_task_baseline(task, model, max_turns=task.max_turns)
            status = "✅" if r.success else "❌"
            print(f"{status} {r.turns_used} turns, {r.total_tokens:,} tok, "
                  f"{r.wall_time_seconds:.1f}s"
                  f"{f', {r.eval_tokens_per_sec:.1f} tok/s' if r.eval_tokens_per_sec else ''}")
        results.append(r)

        # ── Condition B: Naive Cap ──
        print(f"  [B] Naive Cap...", end=" ", flush=True)
        if dry_run:
            r = RunResult(task.name, model, "naive_cap", task.difficulty)
            print("SKIPPED (dry run)")
        else:
            r = run_task_naive_cap(task, model)
            status = "✅" if r.success else "❌"
            print(f"{status} {r.turns_used} turns, {r.total_tokens:,} tok, "
                  f"{r.wall_time_seconds:.1f}s")
        results.append(r)

        # ── Condition C: Harness ──
        print(f"  [C] Harness...", end=" ", flush=True)
        if dry_run:
            r = RunResult(task.name, model, "harness", task.difficulty)
            print("SKIPPED (dry run)")
        else:
            r = run_task_harness(task, model, max_turns=task.max_turns)
            status = "✅" if r.success else "❌"
            trip_info = f" ⚠️ TRIPPED at turn {r.trip_turn}" if r.harness_tripped else ""
            print(f"{status} {r.turns_used} turns, {r.total_tokens:,} tok, "
                  f"{r.wall_time_seconds:.1f}s{trip_info}")
            if r.growth_ratios:
                print(f"      Ratios: {[f'{r:.2f}' for r in r.growth_ratios[-5:]]}")
        results.append(r)

    return results


def print_summary(results: list[RunResult], model: str):
    """Print a summary table of results."""

    print(f"\n\n{'═' * 70}")
    print(f"  RESULTS SUMMARY — {model}")
    print(f"{'═' * 70}\n")

    conditions = ["baseline", "naive_cap", "harness"]

    for difficulty in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r.difficulty == difficulty]
        if not diff_results:
            continue

        print(f"\n  ── {difficulty.upper()} ──\n")
        print(f"  {'Task':<22s} {'Condition':<12s} {'OK?':<5s} {'Turns':<6s} "
              f"{'Tokens':<10s} {'Time':<8s} {'Trip?':<6s} {'Pattern':<20s}")
        print(f"  {'─' * 95}")

        for task_name in sorted(set(r.task_name for r in diff_results)):
            for cond in conditions:
                r = next((r for r in diff_results
                          if r.task_name == task_name and r.condition == cond), None)
                if not r:
                    continue
                ok = "✅" if r.success else "❌"
                trip = f"T{r.trip_turn}" if r.harness_tripped else "—"
                pattern = r.failure_pattern or "—"
                print(f"  {r.task_name:<22s} {r.condition:<12s} {ok:<5s} "
                      f"{r.turns_used:<6d} {r.total_tokens:<10,d} "
                      f"{r.wall_time_seconds:<8.1f} {trip:<6s} {pattern:<20s}")

    # ── Aggregate stats ──
    print(f"\n\n  ── AGGREGATE ──\n")

    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        successes = sum(1 for r in cond_results if r.success)
        total_tokens = sum(r.total_tokens for r in cond_results)
        total_time = sum(r.wall_time_seconds for r in cond_results)
        trips = sum(1 for r in cond_results if r.harness_tripped)
        n = len(cond_results)

        print(f"  {cond:<12s}: {successes}/{n} pass ({100*successes/max(n,1):.1f}%), "
              f"{total_tokens:>10,d} tokens, {total_time:>8.1f}s wall, "
              f"{trips} trips")

    # ── Harness value metrics ──
    print(f"\n\n  ── HARNESS VALUE METRICS ──\n")

    baseline_results = {r.task_name: r for r in results if r.condition == "baseline"}
    harness_results = {r.task_name: r for r in results if r.condition == "harness"}

    tokens_saved = 0
    false_positives = 0  # tripped on a task that baseline succeeded
    true_positives = 0   # tripped on a task that baseline failed
    missed = 0           # didn't trip on a task that baseline failed with high tokens

    for task_name in baseline_results:
        b = baseline_results[task_name]
        h = harness_results.get(task_name)
        if not h:
            continue

        if h.harness_tripped:
            tokens_saved += b.total_tokens - h.tokens_at_trip
            if b.success:
                false_positives += 1
            else:
                true_positives += 1
        elif not b.success and b.total_tokens > 5000:
            missed += 1

    total_baseline_tokens = sum(r.total_tokens for r in baseline_results.values())

    print(f"  Tokens saved by harness:  {tokens_saved:>10,d} "
          f"({100*tokens_saved/max(total_baseline_tokens,1):.1f}% of baseline)")
    print(f"  True positives (caught):  {true_positives}")
    print(f"  False positives (wrong):  {false_positives}")
    print(f"  Missed spirals:           {missed}")
    print()


def save_results(results: list[RunResult], model: str, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_safe = model.replace(":", "_").replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{model_safe}_{timestamp}.json"

    output = {
        "metadata": {
            "model": model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ollama_host": OLLAMA_BASE,
            "num_tasks": len(set(r.task_name for r in results)),
            "num_conditions": 3,
            "total_runs": len(results),
        },
        "guard_config": {
            "token_budget": 100_000,
            "ratio_threshold": 1.8,
            "window": 3,
            "warmup_turns": 2,
            "budget_gate": 3_000,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {output_file}")
    return output_file


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local/edge model benchmark for state-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Ollama model name (e.g., qwen3:4b, llama3.2:3b)",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Run only tasks of this difficulty (default: all)",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Run only a specific task by name",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate benchmark logic without running models",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent.parent.parent / "benchmark_results" / "local_models"),
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Select tasks
    if args.task:
        if args.task not in TASKS_BY_NAME:
            print(f"❌ Unknown task: {args.task}")
            print(f"   Available: {', '.join(TASKS_BY_NAME.keys())}")
            return 1
        tasks = [TASKS_BY_NAME[args.task]]
    elif args.difficulty == "all":
        tasks = ALL_TASKS
    else:
        tasks = TASKS_BY_DIFFICULTY[args.difficulty]

    # Verify Ollama is running
    if not args.dry_run:
        try:
            import urllib.request
            urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5)
        except Exception:
            print(f"❌ Cannot connect to Ollama at {OLLAMA_BASE}")
            print(f"   Start Ollama with: ollama serve")
            return 1

    # Run benchmark
    results = run_benchmark(args.model, tasks, dry_run=args.dry_run)

    if not args.dry_run:
        print_summary(results, args.model)
        save_results(results, args.model, Path(args.output_dir))

    print(f"\n✅ Benchmark complete for {args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
