"""Failure diagnostics showcase.

Demonstrates all detectable failure patterns with realistic
token trajectories. Run this to see what FailureReport outputs
for each pattern.
"""

from state_harness import GrowthRatioGuard, StabilityViolation, FailureReport

SEPARATOR = "=" * 60


def simulate_context_spiral():
    """Agent replaying full history each turn — exponential growth."""
    print(f"\n{SEPARATOR}")
    print("PATTERN: Context Accumulation Spiral")
    print(SEPARATOR)

    guard = GrowthRatioGuard(
        token_budget=200_000,
        ratio_threshold=3.0,  # lenient to let spiral develop
        window=3,
        warmup_turns=3,
        budget_gate=5_000,
    )

    # Realistic trajectory: starts normal, then each turn grows
    # because the agent sends the full conversation as context
    tokens = [1000, 1100, 1050, 1200, 1500, 1800, 2200, 2800, 3500, 4500, 5500, 7000]
    for t in tokens:
        try:
            guard.record_step(tokens_used=t)
        except StabilityViolation:
            break

    print(FailureReport.from_guard(guard, model="gemini-2.5-flash"))


def simulate_retry_storm():
    """Agent stuck retrying the same failed tool call."""
    print(f"\n{SEPARATOR}")
    print("PATTERN: Retry Storm")
    print(SEPARATOR)

    guard = GrowthRatioGuard(
        token_budget=50_000,
        ratio_threshold=2.0,
        window=10,  # wide window to observe the pattern
        warmup_turns=3,
        budget_gate=3_000,
    )

    # 12 identical calls — the agent keeps retrying
    for _ in range(12):
        try:
            guard.record_step(tokens_used=500, errors=1)
        except StabilityViolation:
            break

    # Force tripped state for demo
    guard._inner.is_tripped = True
    print(FailureReport.from_guard(guard, model="gemini-2.5-flash"))


def simulate_early_explosion():
    """Oversized system prompt or massive tool response on turn 1."""
    print(f"\n{SEPARATOR}")
    print("PATTERN: Early Token Explosion")
    print(SEPARATOR)

    guard = GrowthRatioGuard(
        token_budget=100_000,
        ratio_threshold=2.0,
        window=3,
        warmup_turns=3,
        budget_gate=5_000,
    )

    # First turn is massive (malformed prompt or huge tool response)
    tokens = [15000, 12000, 8000, 2000, 1500]
    for t in tokens:
        try:
            guard.record_step(tokens_used=t)
        except StabilityViolation:
            break

    guard._inner.is_tripped = True
    print(FailureReport.from_guard(guard, model="gemini-2.5-flash"))


def simulate_healthy():
    """Normal task that completes within budget."""
    print(f"\n{SEPARATOR}")
    print("PATTERN: Healthy Completion")
    print(SEPARATOR)

    guard = GrowthRatioGuard(
        token_budget=50_000,
        ratio_threshold=2.0,
        window=3,
        warmup_turns=3,
        budget_gate=5_000,
    )

    tokens = [1000, 1100, 1050, 900, 1100, 950, 1000, 1050]
    for t in tokens:
        guard.record_step(tokens_used=t)

    print(FailureReport.from_guard(guard, model="gemini-2.5-flash"))


if __name__ == "__main__":
    print("State-Harness Failure Diagnostics Demo")
    print("Detecting and classifying failure patterns from execution telemetry.\n")

    simulate_context_spiral()
    simulate_retry_storm()
    simulate_early_explosion()
    simulate_healthy()

    print(f"\n{SEPARATOR}")
    print("All patterns demonstrated. No LLM calls were made.")
    print(f"{SEPARATOR}")
