"""State-harness monitored flows for SWE-bench evaluation — Full-Stack v3.

Provides HarnessSearchTree (for SWE-bench's SearchTree flow) and
HarnessLoop (for simpler AgenticLoop flows) that integrate the complete
state-harness safety stack into moatless-tools:

1. GrowthRatioGuard: Normalized token monitoring with adaptive threshold
2. RGDecimator: Compresses accumulated node action text during interventions
3. HolographicEngine: VSA policy drift detection against task description
4. Dual-confirmation gating: Trip only when BOTH growth-ratio AND drift confirm

Feature toggles (via environment variables):
    HARNESS_ENABLED=false     → Disable all monitoring
    HARNESS_RG=off            → Disable RG compression
    HARNESS_VSA=off           → Disable VSA drift detection
"""

import logging
import os
import statistics
from typing import Optional

from pydantic import Field, PrivateAttr

from moatless.flow.loop import AgenticLoop
from moatless.flow.search_tree import SearchTree

from state_harness import (
    GrowthRatioGuard,
    HolographicEngine,
    RGDecimator,
    StabilityViolation,
    BudgetExhausted,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() not in ("off", "0", "false", "no")


class HarnessSearchTree(SearchTree):
    """SearchTree with full-stack state-harness monitoring (v3).

    Drop-in replacement for SearchTree in SWE-bench flow configs.
    Supports 4 experimental conditions via env vars:
      - Baseline: harness_enabled=False
      - Lyapunov-only: HARNESS_RG=off HARNESS_VSA=off
      - Lyapunov+RG: HARNESS_VSA=off
      - Full-stack: (default)
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
    _rg: Optional[RGDecimator] = PrivateAttr(default=None)
    _engine: Optional[HolographicEngine] = PrivateAttr(default=None)
    _rg_enabled: bool = PrivateAttr(default=True)
    _vsa_enabled: bool = PrivateAttr(default=True)
    _intervention_count: int = PrivateAttr(default=0)
    _prev_total_tokens: int = PrivateAttr(default=0)
    _drift_history: list = PrivateAttr(default_factory=list)
    _drift_threshold: float = PrivateAttr(default=0.7)
    _drift_window: int = PrivateAttr(default=3)
    _harness_telemetry: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # Resolve env-var feature toggles
        self._rg_enabled = _env_flag("HARNESS_RG", default=True)
        self._vsa_enabled = _env_flag("HARNESS_VSA", default=True)

        if not _env_flag("HARNESS_ENABLED", default=self.harness_enabled):
            self.harness_enabled = False

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

            if self._rg_enabled:
                self._rg = RGDecimator(threshold=0.3, max_retained=30)

            if self._vsa_enabled:
                self._engine = HolographicEngine(dim=2000)
                # Register task description as invariant
                # (will be set when first node message is available)

            mode = "full-stack" if (self._rg_enabled and self._vsa_enabled) else \
                   "lyapunov+rg" if self._rg_enabled else \
                   "lyapunov+vsa" if self._vsa_enabled else "lyapunov-only"

            logger.info(
                f"[state-harness] HarnessSearchTree v3 initialized: "
                f"mode={mode}, budget={self.harness_budget}, "
                f"adaptive_k={self.harness_adaptive_k}"
            )

    def _register_task_invariant(self):
        """Register the task problem statement as a VSA invariant.

        Called lazily on the first iteration when the root node's
        user_message (containing the problem statement) is available.
        """
        if self._engine is None or self._engine.invariant_count() > 0:
            return

        # Extract task description from root node
        task_text = ""
        if self.root and self.root.user_message:
            task_text = self.root.user_message[:500]
        elif self.root and self.root.action_steps:
            task_text = str(self.root.action_steps[0].action)[:500]

        if task_text:
            key = self._engine.encode_text("swe task objective")
            val = self._engine.encode_text(task_text)
            self._engine.register_invariant("task_objective", key, val)
            logger.debug(
                f"[state-harness] Registered task invariant "
                f"({len(task_text)} chars)"
            )

    def _get_latest_node_text(self) -> str:
        """Extract the latest node's action/observation text for VSA drift."""
        if not self.root:
            return ""

        # Find the deepest leaf node
        node = self.root
        while node.children:
            node = node.children[-1]

        # Get assistant message or action text
        if node.assistant_message:
            return node.assistant_message[:200]
        elif node.action_steps:
            last_step = node.action_steps[-1]
            if last_step.observation:
                return str(last_step.observation)[:200]
            return str(last_step.action)[:200]
        return ""

    def _check_drift_gate(self) -> bool:
        """Check if recent drift confirms instability.

        Returns True if drift is high enough to confirm a trip.
        When VSA is disabled, always returns True (Lyapunov-only mode).
        """
        if not self._vsa_enabled or self._engine is None:
            return True

        if len(self._drift_history) < self._drift_window:
            return True  # Not enough data — conservative

        recent = self._drift_history[-self._drift_window:]
        return statistics.mean(recent) > self._drift_threshold

    def is_finished(self) -> str | None:
        """Check standard conditions + full-stack state-harness monitoring."""
        base_result = super().is_finished()
        if base_result:
            return base_result

        if not self.harness_enabled or not self._guard:
            return None

        # Lazily register task invariant
        self._register_task_invariant()

        # Compute token delta
        usage = self.total_usage()
        current_total = usage.prompt_tokens + usage.completion_tokens
        tokens_this_iter = current_total - self._prev_total_tokens
        self._prev_total_tokens = current_total

        if tokens_this_iter <= 0:
            return None

        # Check VSA drift
        if self._vsa_enabled and self._engine:
            node_text = self._get_latest_node_text()
            if node_text:
                context_vec = self._engine.encode_text(node_text)
                drift = self._engine.check_drift("task_objective", context_vec)
                self._drift_history.append(drift)

        # Record in growth-ratio monitor
        try:
            self._guard.record_step(tokens_used=tokens_this_iter, errors=0)
            logger.debug(
                f"[state-harness] iter tokens={tokens_this_iter}, "
                f"total={current_total}, "
                f"ratio={self._guard.current_ratio or 'warmup'}"
            )
        except StabilityViolation as e:
            # Dual-confirmation: check drift gate
            if self._check_drift_gate():
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
            else:
                # On-policy — suppress trip
                logger.info(
                    f"[state-harness] Growth ratio exceeded but drift is low "
                    f"— suppressing at {current_total} tokens"
                )
        except BudgetExhausted:
            logger.warning(
                f"[state-harness] Budget exhausted at {current_total}"
            )
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
            "baseline": self._guard.baseline if self._guard else None,
            "current_ratio": self._guard.current_ratio if self._guard else None,
            "energy_history": list(self._guard.energy_history) if self._guard else [],
            "drift_history": self._drift_history[-10:],
            "interventions": self._intervention_count,
            "rg_enabled": self._rg_enabled,
            "vsa_enabled": self._vsa_enabled,
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


class HarnessLoop(AgenticLoop):
    """AgenticLoop with full-stack state-harness monitoring (v3)."""

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
    _rg_enabled: bool = PrivateAttr(default=True)
    _vsa_enabled: bool = PrivateAttr(default=True)
    _intervention_count: int = PrivateAttr(default=0)
    _prev_total_tokens: int = PrivateAttr(default=0)
    _harness_telemetry: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self._rg_enabled = _env_flag("HARNESS_RG", default=True)
        self._vsa_enabled = _env_flag("HARNESS_VSA", default=True)

        if not _env_flag("HARNESS_ENABLED", default=self.harness_enabled):
            self.harness_enabled = False

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
