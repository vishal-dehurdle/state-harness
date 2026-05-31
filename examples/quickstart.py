"""Quickstart: Protect your LLM agent in 10 lines.

This is the simplest possible integration. Drop a GrowthRatioGuard
around your agent loop and it will kill runaway spirals automatically.
"""

from state_harness import GrowthRatioGuard, StabilityViolation, FailureReport

# 1. Create a guard
guard = GrowthRatioGuard(
    token_budget=50_000,      # hard ceiling: 50K tokens
    ratio_threshold=2.0,      # trip when a turn uses 2× the baseline
    window=3,                 # 3 consecutive escalating turns → trip
)

# 2. Wrap your agent loop
with guard:
    for turn in range(20):
        # Replace with your actual LLM call:
        #   result = llm.invoke(prompt)
        #   tokens = result.usage.total_tokens
        tokens = 1000 + (turn * 200)  # simulated growing usage

        try:
            guard.record_step(tokens_used=tokens)
        except StabilityViolation as e:
            print(f"🛑 Agent killed at turn {turn}: {e}")
            break

# 3. Get the failure report
report = FailureReport.from_guard(guard, model="gemini-2.5-flash")
print(report)
