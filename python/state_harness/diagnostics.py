# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Failure diagnostics for state-harness.

Analyzes energy trajectories, drift scores, and step history from a
BoundaryGuard or GrowthRatioGuard to produce human-readable failure
reports with pattern classification and actionable suggestions.

No LLM calls, no infrastructure, no cost. Just signal analysis.

Example::

    from state_harness import GrowthRatioGuard
    from state_harness.diagnostics import FailureReport

    guard = GrowthRatioGuard(token_budget=50_000)

    with guard:
        for turn in agent_loop:
            result = llm.invoke(turn)
            guard.record_step(tokens_used=result.usage.total_tokens)

    report = FailureReport.from_guard(guard)
    print(report)
    # or
    print(report.to_dict())  # for structured logging
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union


# ─── Failure Patterns ───────────────────────────────────────────────────────


class FailurePattern(Enum):
    """Classified failure patterns detectable from execution telemetry."""

    CONTEXT_SPIRAL = "context_accumulation_spiral"
    """Token usage grows exponentially as conversation history accumulates
    without compression. The agent is replaying the full context each turn."""

    RETRY_STORM = "retry_storm"
    """Repeated transient errors trigger retries, each consuming tokens
    but making no forward progress."""

    TOOL_LOOP = "tool_loop"
    """The agent calls the same tool repeatedly with identical or similar
    arguments, stuck in a decision loop."""

    POLICY_DRIFT = "policy_drift"
    """The agent's responses drift progressively further from the domain
    policy, indicating loss of task focus."""

    BUDGET_EXHAUSTION = "budget_exhaustion"
    """Cumulative token spend exceeded the budget ceiling. The task was
    expensive but not necessarily spiraling."""

    EARLY_EXPLOSION = "early_explosion"
    """Token usage spikes dramatically in the first few turns, suggesting
    a malformed prompt or excessively large tool response."""

    GRADUAL_DEGRADATION = "gradual_degradation"
    """Slow, steady increase in token usage without clear spiral. The task
    is complex but may eventually complete if given more budget."""

    HEALTHY_COMPLETION = "healthy_completion"
    """No failure detected. The task completed within budget with stable
    energy trajectory."""

    UNKNOWN = "unknown"
    """No clear pattern detected from available signals."""


# ─── Suggestion ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Suggestion:
    """An actionable suggestion for addressing the detected failure pattern.

    Attributes:
        priority: 1 (most urgent) to 3 (nice-to-have).
        action: Short imperative description of what to do.
        rationale: Why this will help.
    """

    priority: int
    action: str
    rationale: str


# ─── Failure Report ─────────────────────────────────────────────────────────


@dataclass
class FailureReport:
    """Structured failure diagnostic report.

    Generated from guard telemetry. Contains the detected failure pattern,
    evidence supporting the classification, cost impact analysis, and
    actionable suggestions for fixing the agent.

    Attributes:
        pattern: The classified failure pattern.
        confidence: Confidence in the classification (0.0–1.0).
        total_tokens: Total tokens consumed.
        total_steps: Number of execution steps.
        is_tripped: Whether the circuit breaker tripped.
        is_frozen: Whether the budget was exhausted.
        baseline_tokens: Median tokens per turn during warmup (if available).
        peak_ratio: Maximum growth ratio observed.
        energy_trajectory: Normalized energy trajectory.
        drift_trajectory: Policy drift scores per turn (if available).
        cost_estimate_usd: Estimated cost in USD (approximate).
        projected_cost_usd: Projected cost if the task had continued.
        suggestions: Ordered list of actionable suggestions.
        evidence: Human-readable evidence strings supporting the diagnosis.
    """

    pattern: FailurePattern
    confidence: float
    total_tokens: int
    total_steps: int
    is_tripped: bool
    is_frozen: bool
    baseline_tokens: Optional[float] = None
    peak_ratio: Optional[float] = None
    energy_trajectory: list[float] = field(default_factory=list)
    drift_trajectory: list[float] = field(default_factory=list)
    cost_estimate_usd: Optional[float] = None
    projected_cost_usd: Optional[float] = None
    suggestions: list[Suggestion] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    # ── Factory Methods ─────────────────────────────────────────────────

    @classmethod
    def from_guard(
        cls,
        guard: Any,
        drift_history: Optional[list[float]] = None,
        model: str = "gemini-2.5-flash",
    ) -> FailureReport:
        """Create a FailureReport from a BoundaryGuard or GrowthRatioGuard.

        Args:
            guard: A ``BoundaryGuard`` or ``GrowthRatioGuard`` instance.
            drift_history: Optional list of drift scores (if not available
                from the guard itself).
            model: Model name for cost estimation. Defaults to gemini-2.5-flash.

        Returns:
            A populated FailureReport with pattern classification.
        """
        # Extract common signals
        total_tokens = guard.total_tokens
        total_steps = guard.total_steps
        is_tripped = guard.is_tripped
        is_frozen = guard.is_frozen
        energy = list(guard.energy_history) if guard.energy_history else []

        # Extract GrowthRatioGuard-specific signals
        baseline = getattr(guard, "baseline", None)
        peak_ratio = None
        if baseline and baseline > 0 and energy:
            # For GrowthRatioGuard, energy_history is already the ratio
            peak_ratio = max(energy) if energy else None

        # Drift history
        drift = drift_history or []

        # Estimate cost
        cost_usd = _estimate_cost(total_tokens, model)
        projected_usd = _project_cost(energy, total_tokens, total_steps, model)

        # Build the report shell
        report = cls(
            pattern=FailurePattern.UNKNOWN,
            confidence=0.0,
            total_tokens=total_tokens,
            total_steps=total_steps,
            is_tripped=is_tripped,
            is_frozen=is_frozen,
            baseline_tokens=baseline,
            peak_ratio=peak_ratio,
            energy_trajectory=energy,
            drift_trajectory=drift,
            cost_estimate_usd=cost_usd,
            projected_cost_usd=projected_usd,
        )

        # Classify the failure pattern
        _classify(report)

        # Generate suggestions
        _suggest(report)

        return report

    # ── Output Formats ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for structured logging / JSON export."""
        return {
            "pattern": self.pattern.value,
            "confidence": round(self.confidence, 2),
            "total_tokens": self.total_tokens,
            "total_steps": self.total_steps,
            "is_tripped": self.is_tripped,
            "is_frozen": self.is_frozen,
            "baseline_tokens": self.baseline_tokens,
            "peak_ratio": round(self.peak_ratio, 2) if self.peak_ratio else None,
            "cost_estimate_usd": round(self.cost_estimate_usd, 4) if self.cost_estimate_usd else None,
            "projected_cost_usd": round(self.projected_cost_usd, 4) if self.projected_cost_usd else None,
            "suggestions": [
                {"priority": s.priority, "action": s.action, "rationale": s.rationale}
                for s in self.suggestions
            ],
            "evidence": self.evidence,
        }

    def __str__(self) -> str:
        """Human-readable failure report for terminal output."""
        lines: list[str] = []

        # Header
        if self.is_tripped or self.is_frozen:
            status = "TRIPPED" if self.is_tripped else "BUDGET EXHAUSTED"
            lines.append(f"⚠️  STABILITY {status} at turn {self.total_steps}")
        else:
            lines.append(f"✅  Completed normally ({self.total_steps} turns)")

        lines.append("")

        # Pattern
        pattern_labels = {
            FailurePattern.CONTEXT_SPIRAL: "Context Accumulation Spiral",
            FailurePattern.RETRY_STORM: "Retry Storm",
            FailurePattern.TOOL_LOOP: "Tool Call Loop",
            FailurePattern.POLICY_DRIFT: "Policy Drift",
            FailurePattern.BUDGET_EXHAUSTION: "Budget Exhaustion",
            FailurePattern.EARLY_EXPLOSION: "Early Token Explosion",
            FailurePattern.GRADUAL_DEGRADATION: "Gradual Degradation",
            FailurePattern.HEALTHY_COMPLETION: "Healthy Completion",
            FailurePattern.UNKNOWN: "Unknown Pattern",
        }
        label = pattern_labels.get(self.pattern, self.pattern.value)
        lines.append(f"Pattern: {label} (confidence: {self.confidence:.0%})")
        lines.append("")

        # Evidence
        if self.evidence:
            for ev in self.evidence:
                lines.append(f"  • {ev}")
            lines.append("")

        # Energy trajectory (compact sparkline)
        if self.energy_trajectory and len(self.energy_trajectory) >= 3:
            sparkline = _sparkline(self.energy_trajectory)
            lines.append(f"Energy: {sparkline}")
            if self.baseline_tokens:
                lines.append(f"  Baseline: {self.baseline_tokens:.0f} tokens/turn")
            if self.peak_ratio:
                lines.append(f"  Peak ratio: {self.peak_ratio:.1f}× baseline")
            lines.append("")

        # Drift trajectory
        if self.drift_trajectory and len(self.drift_trajectory) >= 3:
            sparkline = _sparkline(self.drift_trajectory)
            lines.append(f"Drift:  {sparkline}")
            mean_drift = statistics.mean(self.drift_trajectory[-3:])
            lines.append(f"  Recent avg drift: {mean_drift:.2f}")
            lines.append("")

        # Cost impact
        if self.cost_estimate_usd is not None:
            cost_line = f"Cost: ${self.cost_estimate_usd:.4f}"
            if self.projected_cost_usd and self.projected_cost_usd > self.cost_estimate_usd:
                saved = self.projected_cost_usd - self.cost_estimate_usd
                cost_line += f" (saved ~${saved:.4f} by tripping early)"
            lines.append(cost_line)
            lines.append(f"  Total tokens: {self.total_tokens:,}")
            lines.append("")

        # Suggestions
        if self.suggestions:
            lines.append("Suggested actions:")
            for i, s in enumerate(self.suggestions, 1):
                priority_icon = ["🔴", "🟡", "🟢"][min(s.priority - 1, 2)]
                lines.append(f"  {priority_icon} {i}. {s.action}")
                lines.append(f"     → {s.rationale}")
            lines.append("")

        return "\n".join(lines)


# ─── Pattern Classification ────────────────────────────────────────────────


def _classify(report: FailureReport) -> None:
    """Classify the failure pattern based on available signals."""
    energy = report.energy_trajectory
    drift = report.drift_trajectory
    n = len(energy)

    # No failure
    if not report.is_tripped and not report.is_frozen:
        report.pattern = FailurePattern.HEALTHY_COMPLETION
        report.confidence = 0.95
        report.evidence.append("Task completed within budget without tripping.")
        return

    # Budget exhaustion (frozen but not tripped)
    if report.is_frozen and not report.is_tripped:
        report.pattern = FailurePattern.BUDGET_EXHAUSTION
        report.confidence = 0.9
        report.evidence.append(
            f"Budget ceiling reached after {report.total_tokens:,} tokens."
        )
        return

    if n < 3:
        report.pattern = FailurePattern.UNKNOWN
        report.confidence = 0.3
        report.evidence.append("Too few turns to classify pattern.")
        return

    # ── Check for early explosion (spike in first 3 turns) ──
    if n >= 3 and max(energy[:3]) > 3.0:
        report.pattern = FailurePattern.EARLY_EXPLOSION
        report.confidence = 0.85
        report.evidence.append(
            f"Token usage spiked to {max(energy[:3]):.1f}× in the first 3 turns."
        )
        report.evidence.append(
            "Likely cause: malformed prompt, oversized tool response, or "
            "excessive system prompt."
        )
        return

    # ── Check for context spiral (exponential growth) ──
    if n >= 5:
        # Look at the tail: are the last 5 values consistently > 1.5× baseline?
        tail = energy[-5:]
        if all(v > 1.5 for v in tail):
            # Check if it's accelerating (each value larger than previous)
            increasing = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i - 1])
            if increasing >= 3:
                report.pattern = FailurePattern.CONTEXT_SPIRAL
                report.confidence = min(0.6 + (increasing * 0.08), 0.95)
                report.evidence.append(
                    f"Last {len(tail)} turns all exceeded 1.5× baseline "
                    f"({increasing}/{len(tail)-1} were accelerating)."
                )
                report.evidence.append(
                    f"Peak growth ratio: {max(tail):.1f}× baseline."
                )
                if report.projected_cost_usd and report.cost_estimate_usd:
                    report.evidence.append(
                        f"Without intervention, projected cost was "
                        f"${report.projected_cost_usd:.4f} "
                        f"(actual: ${report.cost_estimate_usd:.4f})."
                    )
                return

    # ── Check for retry storm (many steps, low token variation) ──
    if n >= 5:
        token_cv = _coefficient_of_variation(energy)
        if token_cv < 0.3 and report.total_steps >= 8:
            report.pattern = FailurePattern.RETRY_STORM
            report.confidence = 0.7
            report.evidence.append(
                f"Low token variance (CV={token_cv:.2f}) across {n} turns "
                f"suggests repeated identical calls."
            )
            report.evidence.append(
                "Agent is likely retrying the same operation without "
                "changing its approach."
            )
            return

    # ── Check for policy drift ──
    if drift and len(drift) >= 3:
        recent_drift = statistics.mean(drift[-3:])
        early_drift = statistics.mean(drift[:3]) if len(drift) >= 6 else recent_drift
        drift_increase = recent_drift - early_drift

        if recent_drift > 0.7 and drift_increase > 0.15:
            report.pattern = FailurePattern.POLICY_DRIFT
            report.confidence = min(0.5 + drift_increase, 0.9)
            report.evidence.append(
                f"Policy drift increased from {early_drift:.2f} to "
                f"{recent_drift:.2f} ({drift_increase:+.2f})."
            )
            report.evidence.append(
                "Agent's responses are diverging from the domain policy."
            )
            return

    # ── Check for tool loop (via step history tool names if available) ──
    # This requires step_history with tool names — check if available
    step_history = getattr(
        report, "_step_history", None
    )
    # Fall through: we don't have tool names in the basic report

    # ── Gradual degradation (mild growth, not explosive) ──
    if n >= 5 and any(v > 1.3 for v in energy[-3:]):
        report.pattern = FailurePattern.GRADUAL_DEGRADATION
        report.confidence = 0.6
        report.evidence.append(
            f"Moderate token growth (peak {max(energy):.1f}×) without "
            f"clear exponential pattern."
        )
        report.evidence.append(
            "Task may be genuinely complex. Consider increasing budget "
            "or ratio threshold."
        )
        return

    # ── Fallback ──
    report.pattern = FailurePattern.UNKNOWN
    report.confidence = 0.3
    report.evidence.append(
        "No clear failure pattern detected from available signals."
    )


# ─── Suggestion Generation ─────────────────────────────────────────────────


def _suggest(report: FailureReport) -> None:
    """Generate actionable suggestions based on the failure pattern."""

    if report.pattern == FailurePattern.CONTEXT_SPIRAL:
        report.suggestions = [
            Suggestion(
                priority=1,
                action="Enable RG history compression in your agent loop.",
                rationale=(
                    "The context window is growing unchecked. Compressing "
                    "older messages reduces prompt tokens by 40-60%."
                ),
            ),
            Suggestion(
                priority=2,
                action="Lower the growth ratio threshold to 1.8×.",
                rationale=(
                    "The spiral was detected at "
                    f"{report.peak_ratio:.1f}× baseline. A lower threshold "
                    "would have caught it earlier."
                )
                if report.peak_ratio
                else "Catching spirals earlier reduces wasted tokens.",
            ),
            Suggestion(
                priority=3,
                action="Add a sliding-window context strategy.",
                rationale=(
                    "Instead of sending the full conversation each turn, "
                    "send only the last N messages plus a summary of earlier ones."
                ),
            ),
        ]

    elif report.pattern == FailurePattern.RETRY_STORM:
        report.suggestions = [
            Suggestion(
                priority=1,
                action="Add exponential backoff with jitter to tool calls.",
                rationale=(
                    "The agent is burning tokens on identical retries. "
                    "Backoff prevents tight retry loops."
                ),
            ),
            Suggestion(
                priority=2,
                action="Set a max-retries limit per tool call.",
                rationale=(
                    f"The agent made {report.total_steps} attempts. "
                    "Cap retries at 3 to fail fast."
                ),
            ),
            Suggestion(
                priority=3,
                action="Classify errors as transient vs. permanent.",
                rationale=(
                    "Permanent errors (auth failures, invalid schemas) "
                    "should not be retried at all."
                ),
            ),
        ]

    elif report.pattern == FailurePattern.POLICY_DRIFT:
        report.suggestions = [
            Suggestion(
                priority=1,
                action="Re-inject the domain policy in the system prompt "
                "every N turns.",
                rationale=(
                    "Long conversations cause the model to 'forget' the "
                    "original policy. Periodic re-injection keeps it anchored."
                ),
            ),
            Suggestion(
                priority=2,
                action="Enable dual-confirmation gating in the harness.",
                rationale=(
                    "Trip the circuit breaker only when BOTH growth ratio "
                    "AND drift score confirm failure, reducing false positives."
                ),
            ),
        ]

    elif report.pattern == FailurePattern.EARLY_EXPLOSION:
        report.suggestions = [
            Suggestion(
                priority=1,
                action="Audit your system prompt and first tool response sizes.",
                rationale=(
                    "A token spike in the first 3 turns usually means the "
                    "system prompt or an initial tool response is too large."
                ),
            ),
            Suggestion(
                priority=2,
                action="Truncate or summarize large tool responses before "
                "injecting them into context.",
                rationale=(
                    "Raw database dumps or API responses can be thousands "
                    "of tokens. Summarize before adding to context."
                ),
            ),
        ]

    elif report.pattern == FailurePattern.BUDGET_EXHAUSTION:
        report.suggestions = [
            Suggestion(
                priority=1,
                action="Increase the token budget for this task type.",
                rationale=(
                    f"The task consumed {report.total_tokens:,} tokens "
                    "without spiraling. It may be genuinely complex."
                ),
            ),
            Suggestion(
                priority=2,
                action="Enable history compression to reduce per-turn cost.",
                rationale=(
                    "Even non-spiraling tasks benefit from pruning "
                    "low-relevance messages in long conversations."
                ),
            ),
        ]

    elif report.pattern == FailurePattern.GRADUAL_DEGRADATION:
        report.suggestions = [
            Suggestion(
                priority=2,
                action="Consider increasing the ratio threshold to 2.5×.",
                rationale=(
                    "The growth was moderate, not exponential. A higher "
                    "threshold would let this task complete."
                ),
            ),
            Suggestion(
                priority=3,
                action="Review whether this task type typically needs more turns.",
                rationale=(
                    "Some tasks are genuinely multi-step. Profile typical "
                    "turn counts and set budget accordingly."
                ),
            ),
        ]

    elif report.pattern == FailurePattern.HEALTHY_COMPLETION:
        report.suggestions = []  # No action needed

    else:
        report.suggestions = [
            Suggestion(
                priority=3,
                action="Enable DEBUG logging to capture full execution trace.",
                rationale=(
                    "The failure pattern is unclear from energy/drift signals "
                    "alone. Full traces will reveal the root cause."
                ),
            ),
        ]


# ─── Utility Functions ─────────────────────────────────────────────────────

# Approximate cost per token for common models (USD per token, blended input/output)
_MODEL_COSTS: dict[str, float] = {
    "gemini-2.5-flash": 0.15e-6,  # $0.15 / 1M tokens (blended)
    "gemini-2.5-pro": 1.25e-6,
    "gemini-2.0-flash": 0.10e-6,
    "gpt-4o": 5.0e-6,
    "gpt-4o-mini": 0.15e-6,
    "gpt-5.4": 3.0e-6,
    "claude-sonnet-4": 3.0e-6,
    "claude-3.5-sonnet": 3.0e-6,
}


def _estimate_cost(total_tokens: int, model: str) -> float:
    """Estimate cost in USD from token count and model name."""
    # Try exact match, then substring match
    cost_per_token = _MODEL_COSTS.get(model)
    if cost_per_token is None:
        for key, val in _MODEL_COSTS.items():
            if key in model.lower():
                cost_per_token = val
                break
    if cost_per_token is None:
        cost_per_token = 0.5e-6  # conservative default
    return total_tokens * cost_per_token


def _project_cost(
    energy: list[float],
    total_tokens: int,
    total_steps: int,
    model: str,
    max_projection_steps: int = 20,
) -> Optional[float]:
    """Project what the task would have cost without intervention.

    Uses the energy trajectory growth rate to extrapolate forward.
    """
    if not energy or len(energy) < 3 or total_steps == 0:
        return None

    # Estimate per-turn token rate at the end
    avg_tokens_per_step = total_tokens / total_steps

    # Growth rate: average ratio of last 3 steps
    tail = energy[-3:]
    if tail[0] <= 0:
        return None

    growth_rate = (tail[-1] / tail[0]) ** (1.0 / (len(tail) - 1))

    if growth_rate <= 1.0:
        # Not growing — projected cost ≈ actual cost
        return _estimate_cost(total_tokens, model)

    # Project forward
    projected_tokens = total_tokens
    current_rate = avg_tokens_per_step * energy[-1]
    for _ in range(max_projection_steps):
        current_rate *= growth_rate
        projected_tokens += int(current_rate)
        if projected_tokens > total_tokens * 10:
            break  # cap at 10× to avoid absurd projections

    return _estimate_cost(projected_tokens, model)


def _coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation (std / mean). Lower = less variable."""
    if not values or len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def _sparkline(values: list[float]) -> str:
    """Generate a unicode sparkline from a list of values."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    if mn == mx:
        return blocks[0] * len(values)
    span = mx - mn
    return "".join(
        blocks[min(int((v - mn) / span * (len(blocks) - 1)), len(blocks) - 1)]
        for v in values
    )
