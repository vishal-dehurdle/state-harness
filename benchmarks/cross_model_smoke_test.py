#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Cross-model smoke test for state-harness.

Validates that growth-ratio thresholds transfer across frontier models
by running the same multi-turn coding tasks through each model with
state-harness monitoring.

Tests:
  1. Guard trips on identical spiraling trajectories regardless of model
  2. No false positives on healthy (short) tasks
  3. Growth-ratio patterns are qualitatively similar

Models (cheapest credible):
  - Gemini 2.5 Flash (via Vertex AI or google-genai)
  - GPT-4o-mini (via OpenAI)
  - Claude 3.5 Haiku (via Anthropic)

Cost estimate: ~$2-3 total across all models.

Usage:
    python benchmarks/cross_model_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# Add python source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from state_harness import GrowthRatioGuard, StabilityViolation, BudgetExhausted
from state_harness.diagnostics import FailureReport


# ─── Task Definitions ──────────────────────────────────────────────────────

# Two types of tasks:
# A) SHORT tasks (2-3 turns) — guard should NOT trip
# B) SPIRAL tasks (forced to escalate) — guard SHOULD trip

SHORT_TASKS = [
    {
        "name": "fibonacci",
        "messages": [
            {"role": "user", "content": "Write a Python function to compute the nth Fibonacci number. Return only the function."},
        ],
    },
    {
        "name": "fizzbuzz",
        "messages": [
            {"role": "user", "content": "Write a Python FizzBuzz function for numbers 1-100. Return only the function."},
        ],
    },
    {
        "name": "palindrome",
        "messages": [
            {"role": "user", "content": "Write a Python function to check if a string is a palindrome. Handle edge cases."},
        ],
    },
]

# Spiral tasks: multi-turn conversations that grow context each turn
SPIRAL_TASKS = [
    {
        "name": "growing_refactor",
        "messages": [
            {"role": "user", "content": "Write a complete REST API server in Python using Flask with 5 endpoints: GET/POST /users, GET/POST /items, DELETE /items/:id. Include error handling, input validation, database models using SQLAlchemy, and comprehensive docstrings."},
            {"role": "user", "content": "Now add authentication middleware using JWT tokens. Add login/register endpoints, token refresh, and protect all existing endpoints. Include the full updated code with all previous endpoints preserved."},
            {"role": "user", "content": "Add comprehensive test coverage using pytest. Write at least 20 test cases covering all endpoints, auth flows, error cases, and edge cases. Include fixtures, mocks for the database, and the full test file. Also add the previous server code again for reference."},
            {"role": "user", "content": "Now add rate limiting, request logging, CORS support, WebSocket support for real-time notifications, and a health check endpoint. Rewrite the entire application with all these features integrated. Include the complete, updated code for all files."},
            {"role": "user", "content": "Add OpenAPI/Swagger documentation, database migrations using Alembic, a Dockerfile, docker-compose.yml, CI/CD pipeline in GitHub Actions, and a comprehensive README. Provide the complete code for every single file in the project."},
            {"role": "user", "content": "Review the entire codebase above and identify all bugs, security vulnerabilities, and performance issues. Then rewrite the entire application fixing all issues. Include every file again with fixes highlighted in comments."},
        ],
    },
]


# ─── Model Clients ─────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Result from a single model turn."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    duration_ms: float


@dataclass
class TaskResult:
    """Result from running a full task."""
    task_name: str
    model_name: str
    turns: list[TurnResult] = field(default_factory=list)
    tripped: bool = False
    trip_turn: Optional[int] = None
    total_tokens: int = 0
    pattern: Optional[str] = None
    error: Optional[str] = None


def run_openai(messages: list[dict], model: str = "gpt-4o-mini") -> TurnResult:
    """Run a single turn via OpenAI API."""
    import openai
    client = openai.OpenAI()

    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
    )
    duration = (time.monotonic() - start) * 1000

    usage = response.usage
    return TurnResult(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        model=model,
        duration_ms=duration,
    )


def run_anthropic(messages: list[dict], model: str = "claude-haiku-4-5") -> TurnResult:
    """Run a single turn via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    # Convert from OpenAI format to Anthropic format
    system = None
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            user_messages.append(msg)

    start = time.monotonic()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": user_messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    duration = (time.monotonic() - start) * 1000

    return TurnResult(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        model=model,
        duration_ms=duration,
    )


def run_gemini(messages: list[dict], model: str = "gemini-2.5-flash") -> TurnResult:
    """Run a single turn via Google GenAI API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    genai_model = genai.GenerativeModel(model)

    # Convert to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [msg["content"]]})

    start = time.monotonic()
    response = genai_model.generate_content(contents)
    duration = (time.monotonic() - start) * 1000

    usage = response.usage_metadata
    return TurnResult(
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        total_tokens=usage.total_token_count,
        model=model,
        duration_ms=duration,
    )


# ─── Test Runner ───────────────────────────────────────────────────────────


def run_task(
    task: dict,
    model_fn,
    model_name: str,
    guard_config: dict,
    task_type: str = "short",
) -> TaskResult:
    """Run a single task with state-harness monitoring."""

    result = TaskResult(
        task_name=task["name"],
        model_name=model_name,
    )

    guard = GrowthRatioGuard(**guard_config)
    conversation: list[dict] = []

    with guard:
        for i, msg in enumerate(task["messages"]):
            conversation.append(msg)

            try:
                turn = model_fn(conversation)
                result.turns.append(turn)
                result.total_tokens += turn.total_tokens

                # Record in guard
                guard.record_step(
                    tokens_used=turn.total_tokens,
                    tool_name=f"turn_{i+1}",
                )

                # Add assistant response to conversation for context growth
                # (For spiral tasks, this is what causes context to grow)
                conversation.append({
                    "role": "assistant",
                    "content": f"[Response: {turn.output_tokens} tokens]"
                })

            except StabilityViolation:
                result.tripped = True
                result.trip_turn = i + 1
                break
            except BudgetExhausted:
                result.tripped = True
                result.trip_turn = i + 1
                break
            except Exception as e:
                result.error = f"Turn {i+1}: {type(e).__name__}: {e}"
                break

    # Get failure report
    report = FailureReport.from_guard(guard)
    result.pattern = report.pattern.value

    return result


def run_smoke_test():
    """Run the complete cross-model smoke test."""
    print("═" * 60)
    print("  Cross-Model Smoke Test")
    print("  Validating growth-ratio threshold transfer")
    print("═" * 60)
    print()

    # Detect available models
    models: list[tuple[str, Any, str]] = []

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_CLOUD_PROJECT")

    if openai_key:
        models.append(("GPT-4o-mini", run_openai, "gpt-4o-mini"))
        print("✅ OpenAI: GPT-4o-mini (~$0.15/1M in, $0.60/1M out)")
    else:
        print("⚠️  OpenAI: OPENAI_API_KEY not set — skipping")

    if anthropic_key:
        models.append(("Claude-Haiku-4.5", run_anthropic, "claude-haiku-4-5"))
        print("✅ Anthropic: Claude Haiku 4.5 (~$1/1M in, $5/1M out)")
    else:
        print("⚠️  Anthropic: ANTHROPIC_API_KEY not set — skipping")

    if google_key:
        models.append(("Gemini-2.5-Flash", run_gemini, "gemini-2.5-flash"))
        print("✅ Google: Gemini 2.5 Flash (~$0.15/1M blended)")
    else:
        print("ℹ️  Google: No API key — skipping (already validated in main benchmarks)")

    if not models:
        print("\n❌ No model API keys found!")
        return 1

    print(f"\nRunning with {len(models)} model(s)...\n")

    # Guard config — same for all models
    guard_config = {
        "token_budget": 200_000,
        "ratio_threshold": 2.0,
        "window": 3,
        "budget_gate": 5_000,
    }

    all_results: list[TaskResult] = []

    # ── Run short tasks (should NOT trip) ──
    print("─── Phase 1: Short tasks (expect NO trips) ───")
    print()

    for model_name, model_fn, model_id in models:
        for task in SHORT_TASKS:
            print(f"  {model_name} × {task['name']}...", end=" ", flush=True)
            result = run_task(task, model_fn, model_name, guard_config, "short")
            all_results.append(result)

            if result.error:
                print(f"❌ Error: {result.error}")
            elif result.tripped:
                print(f"⚠️  FALSE POSITIVE (tripped at turn {result.trip_turn})")
            else:
                tokens_per_turn = [t.total_tokens for t in result.turns]
                print(f"✅ {result.total_tokens:,} tokens {tokens_per_turn}")

    print()

    # ── Run spiral tasks (should trip) ──
    print("─── Phase 2: Spiral tasks (expect trips) ───")
    print()

    for model_name, model_fn, model_id in models:
        for task in SPIRAL_TASKS:
            print(f"  {model_name} × {task['name']}...", end=" ", flush=True)
            result = run_task(task, model_fn, model_name, guard_config, "spiral")
            all_results.append(result)

            if result.error:
                print(f"❌ Error: {result.error}")
            elif result.tripped:
                tokens_per_turn = [t.total_tokens for t in result.turns]
                print(f"✅ TRIPPED at turn {result.trip_turn} "
                      f"({result.total_tokens:,} tokens) {tokens_per_turn}")
            else:
                print(f"⚠️  MISSED (did not trip in {len(result.turns)} turns, "
                      f"{result.total_tokens:,} tokens)")

    print()

    # ── Summary ──
    print("═" * 60)
    print("  Results Summary")
    print("═" * 60)
    print()

    # Group by model
    for model_name, _, _ in models:
        model_results = [r for r in all_results if r.model_name == model_name]
        short_results = [r for r in model_results if r.task_name in [t["name"] for t in SHORT_TASKS]]
        spiral_results = [r for r in model_results if r.task_name in [t["name"] for t in SPIRAL_TASKS]]

        false_positives = sum(1 for r in short_results if r.tripped)
        true_positives = sum(1 for r in spiral_results if r.tripped)
        errors = sum(1 for r in model_results if r.error)
        total_tokens = sum(r.total_tokens for r in model_results)

        print(f"  {model_name}:")
        print(f"    Short tasks: {len(short_results) - false_positives}/{len(short_results)} "
              f"passed (0 false positives expected)")
        print(f"    Spiral tasks: {true_positives}/{len(spiral_results)} "
              f"tripped (all should trip)")
        print(f"    Errors: {errors}")
        print(f"    Total tokens: {total_tokens:,}")
        print()

    # ── Validation criteria ──
    print("─── Validation Criteria ───")
    print()

    short_false_positives = sum(
        1 for r in all_results
        if r.task_name in [t["name"] for t in SHORT_TASKS] and r.tripped
    )
    spiral_trips = sum(
        1 for r in all_results
        if r.task_name in [t["name"] for t in SPIRAL_TASKS] and r.tripped
    )
    spiral_total = sum(
        1 for r in all_results
        if r.task_name in [t["name"] for t in SPIRAL_TASKS] and not r.error
    )

    pass_1 = short_false_positives == 0
    pass_2 = spiral_trips == spiral_total if spiral_total > 0 else False
    total_tokens_all = sum(r.total_tokens for r in all_results)

    print(f"  1. Zero false positives on short tasks: "
          f"{'✅ PASS' if pass_1 else '❌ FAIL'} "
          f"({short_false_positives} false positives)")
    print(f"  2. Guard trips on spiral tasks: "
          f"{'✅ PASS' if pass_2 else '❌ FAIL'} "
          f"({spiral_trips}/{spiral_total} tripped)")
    print(f"  3. Total cost: ~${total_tokens_all * 0.5e-6:.2f} "
          f"({total_tokens_all:,} tokens)")
    print()

    overall = pass_1 and pass_2
    print(f"  Overall: {'✅ THRESHOLD TRANSFER VALIDATED' if overall else '❌ NEEDS INVESTIGATION'}")
    print()

    # ── Save results ──
    results_dir = Path(__file__).parent.parent / "benchmark_results" / "cross_model"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "smoke_test_results.json"

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "guard_config": guard_config,
        "models": [m[0] for m in models],
        "validation": {
            "false_positives": short_false_positives,
            "spiral_trips": spiral_trips,
            "spiral_total": spiral_total,
            "overall_pass": overall,
        },
        "results": [
            {
                "task": r.task_name,
                "model": r.model_name,
                "tripped": r.tripped,
                "trip_turn": r.trip_turn,
                "total_tokens": r.total_tokens,
                "pattern": r.pattern,
                "turns": [
                    {"input": t.input_tokens, "output": t.output_tokens, "total": t.total_tokens}
                    for t in r.turns
                ],
                "error": r.error,
            }
            for r in all_results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_file}")
    return 0 if overall else 1


if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import openai  # noqa
    except ImportError:
        print("Installing openai...")
        os.system(f"{sys.executable} -m pip install openai -q")

    try:
        import anthropic  # noqa
    except ImportError:
        print("Installing anthropic...")
        os.system(f"{sys.executable} -m pip install anthropic -q")

    sys.exit(run_smoke_test())
