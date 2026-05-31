"""State-harness monitored flows for SWE-bench evaluation.

Provides HarnessSearchTree (for SWE-bench's SearchTree flow) and
HarnessLoop (for simpler AgenticLoop flows) that integrate Lyapunov
growth-ratio monitoring into moatless-tools.

After each iteration, records cumulative token usage into GrowthRatioGuard.
When the monitor detects a spiral (escalating growth ratio exceeding
adaptive threshold), it terminates the flow early to save cost.
"""

import logging
from typing import Optional

from pydantic import Field, PrivateAttr

from moatless.flow.loop import AgenticLoop
from moatless.flow.search_tree import SearchTree

from state_harness import (
    GrowthRatioGuard,
    StabilityViolation,
    BudgetExhausted,
)

logger = logging.getLogger(__name__)


class _HarnessMixin:
    """Mixin providing state-harness monitoring for any moatless flow.

    Tracks per-iteration token deltas via GrowthRatioGuard and checks
    for stability violations in is_finished().
    """

    harness_budget: int = 300_000
    harness_ratio_threshold: float = 2.0
    harness_warmup: int = 5
    harness_window: int = 3
    harness_budget_gate: int = 20_000
    harness_adaptive: bool = True
    harness_adaptive_k: float = 1.0
    harness_max_interventions: int = 2
    harness_enabled: bool = True

    def _init_harness(self):
        """Initialize the GrowthRatioGuard. Call from model_post_init."""
        self._guard: Optional[GrowthRatioGuard] = None
        self._intervention_count: int = 0
        self._prev_total_tokens: int = 0
        self._harness_telemetry: dict = {}

        if self.harness_enabled:
            self._guard = GrowthRatioGuard(
                token_budget=self.harness_budget,
                ratio_threshold=self.harness_ratio_threshold,
                window=self.harness_window,
                warmup_turns=self.harness_warmup,
                budget_gate=self.harness_budget_gate,
                adaptive=self.harness_adaptive,
                adaptive_k=self.harness_adaptive_k,
            )
            logger.info(
                f"[state-harness] Initialized: budget={self.harness_budget}, "
                f"adaptive={self.harness_adaptive}, k={self.harness_adaptive_k}"
            )

    def _harness_check(self) -> str | None:
        """Run growth-ratio check. Returns finish reason or None."""
        if not self.harness_enabled or not self._guard:
            return None

        usage = self.total_usage()
        current_total = usage.prompt_tokens + usage.completion_tokens
        tokens_this_iter = current_total - self._prev_total_tokens
        self._prev_total_tokens = current_total

        if tokens_this_iter <= 0:
            return None

        try:
            self._guard.record_step(tokens_used=tokens_this_iter, errors=0)
            ratio = self._guard.current_ratio
            logger.debug(
                f"[state-harness] tokens={tokens_this_iter}, "
                f"total={current_total}, ratio={ratio or 'warmup'}"
            )
        except StabilityViolation as e:
            if self._intervention_count >= self.harness_max_interventions:
                logger.warning(
                    f"[state-harness] KILLED at {current_total} tokens "
                    f"after {self._intervention_count} interventions: {e}"
                )
                self._harness_telemetry = self._build_telemetry(
                    current_total, "stability_violation"
                )
                return "harness_stability_violation"
            else:
                self._intervention_count += 1
                self._guard.reset_escalation()
                logger.info(
                    f"[state-harness] Intervention "
                    f"#{self._intervention_count}/{self.harness_max_interventions}"
                )
        except BudgetExhausted:
            logger.warning(f"[state-harness] Budget exhausted at {current_total}")
            self._harness_telemetry = self._build_telemetry(
                current_total, "budget_exhausted"
            )
            return "harness_budget_exhausted"

        return None

    def _build_telemetry(self, total_tokens: int, reason: str) -> dict:
        """Build telemetry dict for post-mortem analysis."""
        return {
            "total_tokens": total_tokens,
            "termination_reason": reason,
            "is_stable": self._guard.is_stable if self._guard else None,
            "is_tripped": self._guard.is_tripped if self._guard else None,
            "baseline": self._guard.baseline if self._guard else None,
            "current_ratio": self._guard.current_ratio if self._guard else None,
            "energy_history": (
                list(self._guard.energy_history) if self._guard else []
            ),
            "intervention_count": self._intervention_count,
        }

    @property
    def harness_telemetry(self) -> dict:
        if self._harness_telemetry:
            return self._harness_telemetry
        if not self._guard:
            return {}
        usage = self.total_usage()
        total = usage.prompt_tokens + usage.completion_tokens
        return self._build_telemetry(total, "completed_normally")


class HarnessSearchTree(SearchTree):
    """SearchTree with state-harness Lyapunov monitoring.

    Drop-in replacement for SearchTree in SWE-bench flow configs.
    Just change flow_class to moatless.flow.harness_loop.HarnessSearchTree.
    """

    harness_budget: int = Field(default=300_000)
    harness_ratio_threshold: float = Field(default=2.0)
    harness_warmup: int = Field(default=5)
    harness_window: int = Field(default=3)
    harness_budget_gate: int = Field(default=20_000)
    harness_adaptive: bool = Field(default=True)
    harness_adaptive_k: float = Field(default=1.0)
    harness_max_interventions: int = Field(default=2)
    harness_enabled: bool = Field(default=True)

    _guard: Optional[GrowthRatioGuard] = PrivateAttr(default=None)
    _intervention_count: int = PrivateAttr(default=0)
    _prev_total_tokens: int = PrivateAttr(default=0)
    _harness_telemetry: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.harness_enabled:
            self._guard = GrowthRatioGuard(
                token_budget=self.harness_budget,
                ratio_threshold=self.harness_ratio_threshold,
                window=self.harness_window,
                warmup_turns=self.harness_warmup,
                budget_gate=self.harness_budget_gate,
                adaptive=self.harness_adaptive,
                adaptive_k=self.harness_adaptive_k,
            )
            logger.info(
                f"[state-harness] HarnessSearchTree initialized: "
                f"budget={self.harness_budget}, adaptive_k={self.harness_adaptive_k}"
            )

    def is_finished(self) -> str | None:
        """Check standard conditions + state-harness monitoring."""
        base_result = super().is_finished()
        if base_result:
            return base_result

        # Run harness check
        if not self.harness_enabled or not self._guard:
            return None

        usage = self.total_usage()
        current_total = usage.prompt_tokens + usage.completion_tokens
        tokens_this_iter = current_total - self._prev_total_tokens
        self._prev_total_tokens = current_total

        if tokens_this_iter <= 0:
            return None

        try:
            self._guard.record_step(tokens_used=tokens_this_iter, errors=0)
            logger.debug(
                f"[state-harness] iter tokens={tokens_this_iter}, "
                f"total={current_total}, "
                f"ratio={self._guard.current_ratio or 'warmup'}"
            )
        except StabilityViolation as e:
            if self._intervention_count >= self.harness_max_interventions:
                logger.warning(
                    f"[state-harness] KILLED at {current_total} tokens: {e}"
                )
                self._harness_telemetry = {
                    "total_tokens": current_total,
                    "termination_reason": "stability_violation",
                    "interventions": self._intervention_count,
                    "baseline": self._guard.baseline,
                    "current_ratio": self._guard.current_ratio,
                    "energy_history": list(self._guard.energy_history),
                }
                return "harness_stability_violation"
            else:
                self._intervention_count += 1
                self._guard.reset_escalation()
                logger.info(
                    f"[state-harness] Intervention "
                    f"#{self._intervention_count}/{self.harness_max_interventions}"
                )
        except BudgetExhausted:
            logger.warning(f"[state-harness] Budget exhausted at {current_total}")
            self._harness_telemetry = {
                "total_tokens": current_total,
                "termination_reason": "budget_exhausted",
            }
            return "harness_budget_exhausted"

        return None

    @property
    def harness_telemetry(self) -> dict:
        if self._harness_telemetry:
            return self._harness_telemetry
        if not self._guard:
            return {}
        usage = self.total_usage()
        total = usage.prompt_tokens + usage.completion_tokens
        return {
            "total_tokens": total,
            "termination_reason": "completed_normally",
            "baseline": self._guard.baseline,
            "current_ratio": self._guard.current_ratio,
            "energy_history": list(self._guard.energy_history),
            "interventions": self._intervention_count,
        }


class HarnessLoop(AgenticLoop):
    """AgenticLoop with state-harness monitoring. For non-SWE-bench flows."""

    harness_budget: int = Field(default=300_000)
    harness_ratio_threshold: float = Field(default=2.0)
    harness_warmup: int = Field(default=5)
    harness_window: int = Field(default=3)
    harness_budget_gate: int = Field(default=20_000)
    harness_adaptive: bool = Field(default=True)
    harness_adaptive_k: float = Field(default=1.0)
    harness_max_interventions: int = Field(default=2)
    harness_enabled: bool = Field(default=True)

    _guard: Optional[GrowthRatioGuard] = PrivateAttr(default=None)
    _intervention_count: int = PrivateAttr(default=0)
    _prev_total_tokens: int = PrivateAttr(default=0)
    _harness_telemetry: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.harness_enabled:
            self._guard = GrowthRatioGuard(
                token_budget=self.harness_budget,
                ratio_threshold=self.harness_ratio_threshold,
                window=self.harness_window,
                warmup_turns=self.harness_warmup,
                budget_gate=self.harness_budget_gate,
                adaptive=self.harness_adaptive,
                adaptive_k=self.harness_adaptive_k,
            )

    def is_finished(self) -> str | None:
        base_result = super().is_finished()
        if base_result:
            return base_result

        if not self.harness_enabled or not self._guard:
            return None

        usage = self.total_usage()
        current_total = usage.prompt_tokens + usage.completion_tokens
        tokens_this_iter = current_total - self._prev_total_tokens
        self._prev_total_tokens = current_total

        if tokens_this_iter <= 0:
            return None

        try:
            self._guard.record_step(tokens_used=tokens_this_iter, errors=0)
        except StabilityViolation:
            if self._intervention_count >= self.harness_max_interventions:
                return "harness_stability_violation"
            self._intervention_count += 1
            self._guard.reset_escalation()
        except BudgetExhausted:
            return "harness_budget_exhausted"

        return None
