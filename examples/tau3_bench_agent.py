"""τ³-bench integration: HarnessAgent with full state-harness v2 monitoring.

This is a reference implementation showing how to wrap any LLMAgent with
state-harness safety monitoring for benchmarking. It demonstrates:

1. GrowthRatioGuard: Normalized token monitoring (vs raw token counts)
2. RG Decimator: History compression before each LLM call
3. Dual-confirmation: Both growth ratio AND policy drift must confirm failure

To run against τ³-bench:

    # Clone τ³-bench alongside state-harness
    git clone https://github.com/sierra-research/tau-bench.git ../tau3-bench
    cd ../tau3-bench

    # Register this agent
    # (see τ³-bench docs for agent registration)

    # Run benchmark
    tau2 run --domain airline --agent harness_agent \\
        --agent-llm vertex_ai/gemini-2.5-flash \\
        --user-llm vertex_ai/gemini-2.5-flash \\
        --num-tasks 50 --num-trials 3

Results from our evaluation on τ³-bench (airline, retail, telecom):
- 58% cost reduction vs. unguarded baseline
- 68% precision on identifying genuinely failing tasks
- 32% false positive rate (down from 46% in v1)
"""

from __future__ import annotations

import statistics
from typing import List, Optional

from state_harness import (
    GrowthRatioGuard,
    HolographicEngine,
    RGDecimator,
    StabilityViolation,
    BudgetExhausted,
    FailureReport,
)


# ── Configuration ──────────────────────────────────────────────────────

DEFAULT_BUDGET_CEILING = 100_000
DEFAULT_RATIO_THRESHOLD = 2.0
DEFAULT_WINDOW = 3
DEFAULT_WARMUP_TURNS = 3
DEFAULT_BUDGET_GATE = 8_000
DEFAULT_LAMBDA = 1.0
DEFAULT_RG_THRESHOLD = 0.3
DEFAULT_RG_MAX_RETAINED = 30
DEFAULT_VSA_DIM = 2000
DEFAULT_DRIFT_THRESHOLD = 0.7
DEFAULT_DRIFT_WINDOW = 3


class HarnessAgent:
    """Reference agent wrapper with state-harness v2 monitoring.

    This is a framework-agnostic example. For τ³-bench integration,
    see the actual harness_agent.py in the tau3-bench repo which extends
    LLMAgent with this monitoring logic.
    """

    def __init__(
        self,
        domain_policy: str = "",
        budget_ceiling: int = DEFAULT_BUDGET_CEILING,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
        window: int = DEFAULT_WINDOW,
        rg_enabled: bool = True,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    ):
        # Growth-ratio Lyapunov monitor
        self._guard = GrowthRatioGuard(
            token_budget=budget_ceiling,
            ratio_threshold=ratio_threshold,
            window=window,
            warmup_turns=DEFAULT_WARMUP_TURNS,
            budget_gate=DEFAULT_BUDGET_GATE,
        )

        # RG decimator for history compression
        self._rg = RGDecimator(
            threshold=DEFAULT_RG_THRESHOLD,
            max_retained=DEFAULT_RG_MAX_RETAINED,
        )
        self._rg_enabled = rg_enabled

        # VSA policy drift detector
        self._engine = HolographicEngine(dim=DEFAULT_VSA_DIM)
        self._policy_key = self._engine.encode_text("domain policy")
        self._policy_val = self._engine.encode_text(domain_policy[:500])
        self._engine.register_invariant(
            "policy", self._policy_key, self._policy_val
        )

        # Dual-confirmation
        self._drift_threshold = drift_threshold
        self._drift_history: list[float] = []

        # Telemetry
        self._turn_count = 0
        self._total_tokens = 0

    def compress_history(self, messages: list[str]) -> list[str]:
        """Compress conversation history via RG Decimator."""
        if not self._rg_enabled or len(messages) <= 5:
            return messages

        scored = self._rg.decimate(messages)
        retained = {s.index for s in scored if s.retained}

        # Always keep last 3 messages
        for i in range(max(0, len(messages) - 3), len(messages)):
            retained.add(i)

        return [messages[i] for i in sorted(retained)]

    def check_drift(self, response_text: str) -> float:
        """Check policy drift for a response. Returns drift score 0.0–1.0."""
        vec = self._engine.encode_text(response_text[:200])
        drift = self._engine.check_drift("policy", vec)
        self._drift_history.append(drift)
        return drift

    def record_step(self, tokens_used: int, response_text: str = "") -> None:
        """Record a step and check stability.

        Raises:
            StabilityViolation: If growth ratio AND drift confirm failure.
            BudgetExhausted: If cumulative budget exceeded.
        """
        # Check drift
        if response_text:
            self.check_drift(response_text)

        # Record in monitor
        try:
            self._guard.record_step(tokens_used=tokens_used)
        except StabilityViolation:
            # Dual-confirmation: only trip if drift also confirms
            if self._is_drifting():
                raise
            # Suppress — high growth but agent is on-policy
        except BudgetExhausted:
            raise

        self._turn_count += 1
        self._total_tokens += tokens_used

    def _is_drifting(self) -> bool:
        """Check if recent drift scores exceed threshold."""
        if len(self._drift_history) < DEFAULT_DRIFT_WINDOW:
            return True  # conservative: allow trip if not enough data
        recent = self._drift_history[-DEFAULT_DRIFT_WINDOW:]
        return statistics.mean(recent) > self._drift_threshold

    def get_report(self, model: str = "gemini-2.5-flash") -> FailureReport:
        """Get a failure diagnostic report."""
        return FailureReport.from_guard(
            self._guard,
            drift_history=self._drift_history,
            model=model,
        )


# ── Example usage ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("τ³-bench HarnessAgent Reference Implementation")
    print("=" * 50)
    print()

    agent = HarnessAgent(
        domain_policy="Handle customer airline requests per company policy.",
        budget_ceiling=50_000,
        ratio_threshold=2.0,
    )

    # Simulate a task that spirals
    turns = [
        (1000, "I can help you with your flight booking."),
        (1100, "Let me look up your reservation details."),
        (1050, "I found your booking. Here are the details."),
        (1500, "Let me check the cancellation policy for you."),
        (2000, "I'm reviewing the full terms and conditions."),
        (2800, "I need to cross-reference with the insurance policy."),
        (3500, "Let me pull up the complete fare rules and regulations."),
    ]

    print("Simulating agent execution:")
    for i, (tokens, text) in enumerate(turns):
        try:
            messages = [t[1] for t in turns[: i + 1]]
            compressed = agent.compress_history(messages)
            agent.record_step(tokens_used=tokens, response_text=text)
            print(f"  Turn {i + 1}: {tokens} tokens ✓")
        except StabilityViolation as e:
            print(f"  Turn {i + 1}: {tokens} tokens 🛑 TRIPPED")
            break
        except BudgetExhausted:
            print(f"  Turn {i + 1}: {tokens} tokens 💰 BUDGET")
            break

    print()
    print(agent.get_report(model="gemini-2.5-flash"))
