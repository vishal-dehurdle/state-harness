# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""User-facing SDK layer providing decorators and context managers
for instrumenting agent execution loops with Lyapunov stability monitoring.

The SDK wraps the Rust compute engine (``state_harness._core``) in an
idiomatic Python API, offering two integration surfaces:

1. **Context Manager** (``BoundaryGuard``): Explicit control over the
   monitoring lifecycle with manual ``record_step()`` calls.

2. **Decorator** (``@boundary_guard``): Automatic per-invocation monitoring
   with pluggable token extraction via ``token_counter`` callbacks.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

from state_harness._core import (
    BudgetExhausted,
    LyapunovMonitor,
    PermanentFailure,
    StabilityStatus,
    StabilityViolation,
)

T = TypeVar("T")


# ─── Protocols ──────────────────────────────────────────────────────────────


@runtime_checkable
class CoarseGrainer(Protocol):
    """Protocol for custom RG coarse-graining implementations.

    Implement this protocol to provide a custom message compression strategy
    (e.g., LLM-driven summarization) that can replace or augment the default
    Rust algebraic decimator.

    The default ``RGDecimator`` (Rust core) uses statistical scoring (TF-IDF
    keyword density + VSA cosine similarity). Implement ``CoarseGrainer`` to
    wire in an LLM summarizer or embedding-based approach if you want higher
    quality compression at the cost of latency.

    .. warning::

        If your implementation calls an LLM, be aware that the summarization
        itself consumes tokens and is subject to the same drift/tsunami risks.
        Budget summarization calls separately from the main agent loop.

    Example::

        class LLMCoarseGrainer:
            def __init__(self, llm_client):
                self.llm = llm_client

            def compress(self, messages: list[str]) -> list[str]:
                summary = self.llm.invoke(
                    f"Summarize these agent messages into key actions: {messages}"
                )
                return [summary]

            def score(self, message: str, context: list[str]) -> float:
                # Return a relevance score in [0.0, 1.0]
                return 0.5  # placeholder
    """

    def compress(self, messages: list[str]) -> list[str]:
        """Compress a list of messages into a reduced set.

        Args:
            messages: Full conversation history (list of message strings).

        Returns:
            Compressed list of message strings in retained order.
        """
        ...

    def score(self, message: str, context: list[str]) -> float:
        """Score a single message for relevance against its context.

        Args:
            message: The message to evaluate.
            context: Preceding messages for context-aware scoring.

        Returns:
            Relevance score in [0.0, 1.0]. Higher = more relevant.
        """
        ...


# ─── Data Classes ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepMetrics:
    """Metrics captured for a single execution step.

    Attributes:
        tokens_used: Number of tokens consumed in this step.
        errors: Number of transient errors/retries in this step.
        latency_ms: Wall-clock time for this step in milliseconds.
        tool_name: Name of the tool or function that was called.
        metadata: Arbitrary additional metadata for diagnostics.
    """

    tokens_used: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    tool_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardConfig:
    """Configuration for the BoundaryGuard.

    Attributes:
        token_budget: Maximum cumulative tokens before budget exhaustion.
        lambda_: Coupling constant λ for error weighting in V(k).
        window: Consecutive non-negative ΔV steps before circuit breaker trips.
        on_violation: Optional callback invoked on stability violation.
        on_budget_exhausted: Optional callback invoked on budget exhaustion.
        on_permanent_failure: Optional callback invoked on permanent failure.
    """

    token_budget: int = 100_000
    lambda_: float = 1.0
    window: int = 3
    on_violation: Optional[Callable[[StabilityViolation], None]] = None
    on_budget_exhausted: Optional[Callable[[BudgetExhausted], None]] = None
    on_permanent_failure: Optional[Callable[[PermanentFailure], None]] = None


class FailureType(Enum):
    """Classification of execution failures for routing to the correct handler."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    BUDGET = "budget"


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for extracting token counts from execution results.

    Implement this protocol to plug custom token extraction logic into
    the ``@boundary_guard`` decorator or ``VanillaHook`` adapter.

    Example::

        def count_openai_tokens(response) -> int:
            return response.usage.total_tokens
    """

    def __call__(self, result: Any) -> int: ...


# ─── BoundaryGuard Context Manager ─────────────────────────────────────────


class BoundaryGuard:
    """Context manager for Lyapunov-stabilized agent execution.

    Wraps agent execution loops and continuously monitors the energy
    derivative ΔV to detect and intercept runaway execution patterns.

    The guard maintains a ``LyapunovMonitor`` on the Rust side and records
    step-level metrics via ``record_step()``. When ΔV ≥ 0 for ``window``
    consecutive steps, the circuit breaker trips with a ``StabilityViolation``.

    Example::

        with BoundaryGuard(token_budget=50_000) as guard:
            for turn in agent_loop:
                result = llm.invoke(turn.prompt)
                guard.record_step(
                    tokens_used=result.usage.total_tokens,
                    errors=turn.retry_count,
                    tool_name=turn.tool_name,
                )
    """

    def __init__(
        self,
        token_budget: int = 100_000,
        lambda_: float = 1.0,
        window: int = 3,
        on_violation: Optional[Callable[[StabilityViolation], None]] = None,
        on_budget_exhausted: Optional[Callable[[BudgetExhausted], None]] = None,
        on_permanent_failure: Optional[Callable[[PermanentFailure], None]] = None,
    ) -> None:
        self._config = GuardConfig(
            token_budget=token_budget,
            lambda_=lambda_,
            window=window,
            on_violation=on_violation,
            on_budget_exhausted=on_budget_exhausted,
            on_permanent_failure=on_permanent_failure,
        )
        self._monitor = LyapunovMonitor(
            lambda_=lambda_,
            window=window,
            budget_ceiling=token_budget,
        )
        self._step_history: list[StepMetrics] = []
        self._start_time: Optional[float] = None
        self._active = False

    # ── Context Manager Protocol ────────────────────────────────────────

    def __enter__(self) -> BoundaryGuard:
        self._start_time = time.monotonic()
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        self._active = False
        # Never suppress exceptions — let StabilityViolation et al. propagate
        return False

    # ── Primary API ─────────────────────────────────────────────────────

    def record_step(
        self,
        tokens_used: int,
        errors: int = 0,
        tool_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StabilityStatus:
        """Record metrics for a single execution step and check stability.

        Args:
            tokens_used: Tokens consumed in this step (instantaneous).
            errors: Number of transient errors/retries in this step.
            tool_name: Optional name of the tool or function called.
            metadata: Optional additional metadata for post-mortem diagnostics.

        Returns:
            Current stability status after this step.

        Raises:
            StabilityViolation: If ΔV ≥ 0 for ``window`` consecutive steps.
            BudgetExhausted: If cumulative tokens exceed the budget ceiling.
        """
        elapsed_ms = 0.0
        if self._start_time is not None:
            elapsed_ms = (time.monotonic() - self._start_time) * 1000.0

        step = StepMetrics(
            tokens_used=tokens_used,
            errors=errors,
            latency_ms=elapsed_ms,
            tool_name=tool_name,
            metadata=metadata or {},
        )
        self._step_history.append(step)

        try:
            status = self._monitor.record_step(tokens_used, errors)
            return status
        except StabilityViolation as exc:
            if self._config.on_violation is not None:
                self._config.on_violation(exc)
                return StabilityStatus.Tripped
            raise
        except BudgetExhausted as exc:
            if self._config.on_budget_exhausted is not None:
                self._config.on_budget_exhausted(exc)
                return StabilityStatus.Frozen
            raise

    def report_transient(self) -> StabilityStatus:
        """Report a transient failure (network timeout, rate limit).

        Increments the error counter by +1.0 within the safety envelope.
        Equivalent to ``record_step(tokens_used=0, errors=1)``.

        Returns:
            Current stability status.

        Raises:
            StabilityViolation: If the transient failure pushes ΔV past the window.
        """
        try:
            return self._monitor.report_transient_failure()
        except StabilityViolation as exc:
            if self._config.on_violation is not None:
                self._config.on_violation(exc)
                return StabilityStatus.Tripped
            raise

    def report_permanent(self, reason: str) -> None:
        """Report a permanent failure (schema violation, invalid configuration).

        Immediately trips the circuit breaker. Post-mortem telemetry is compiled
        into the exception message before propagation.

        Args:
            reason: Human-readable description of the permanent failure.

        Raises:
            PermanentFailure: Always raised (unless suppressed by callback).
        """
        try:
            self._monitor.report_permanent_failure(reason)
        except PermanentFailure as exc:
            if self._config.on_permanent_failure is not None:
                self._config.on_permanent_failure(exc)
            else:
                raise

    def check(self) -> StabilityStatus:
        """Check current stability without recording new metrics."""
        return self._monitor.check_stability()

    # ── Introspection Properties ────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all recorded steps."""
        return self._monitor.total_tokens()

    @property
    def total_steps(self) -> int:
        """Total number of recorded steps."""
        return self._monitor.total_steps()

    @property
    def energy_history(self) -> list[float]:
        """Full V(k) energy trajectory for diagnostics and visualization."""
        return self._monitor.get_energy_history()

    @property
    def step_history(self) -> list[StepMetrics]:
        """Full history of recorded step metrics."""
        return list(self._step_history)

    @property
    def is_stable(self) -> bool:
        """Whether the system is currently in a stable or warning state."""
        status = self._monitor.check_stability()
        return status in (StabilityStatus.Stable, StabilityStatus.Warning)

    @property
    def is_tripped(self) -> bool:
        """Whether the circuit breaker has been tripped."""
        return self._monitor.is_tripped()

    @property
    def is_frozen(self) -> bool:
        """Whether the state has been frozen due to budget exhaustion."""
        return self._monitor.is_frozen()

    @property
    def snapshot(self) -> Any:
        """Compile a frozen telemetry snapshot for post-mortem analysis."""
        return self._monitor.snapshot()


# ─── @boundary_guard Decorator ──────────────────────────────────────────────


def boundary_guard(
    token_budget: int = 100_000,
    lambda_: float = 1.0,
    window: int = 3,
    token_counter: Optional[TokenCounter] = None,
    on_violation: Optional[Callable[[StabilityViolation], None]] = None,
    on_budget_exhausted: Optional[Callable[[BudgetExhausted], None]] = None,
) -> Callable:
    """Decorator for Lyapunov-stabilized function execution.

    Wraps a function (typically an agent step or LLM invocation) with automatic
    stability monitoring. Each call to the decorated function records one step
    in the Lyapunov monitor with token usage extracted via ``token_counter``.

    A single ``BoundaryGuard`` instance is shared across all invocations of the
    decorated function, tracking cumulative energy across the execution lifecycle.

    Args:
        token_budget: Maximum total tokens allowed across all invocations.
        lambda_: Coupling constant for error weighting in V(k).
        window: Consecutive non-negative ΔV steps before tripping.
        token_counter: Callable that extracts token count from the function's
                       return value. If None, tokens default to 0 per step.
        on_violation: Callback invoked on stability violation (suppresses exception).
        on_budget_exhausted: Callback invoked on budget exhaustion.

    Returns:
        Decorator function.

    Example::

        @boundary_guard(
            token_budget=50_000,
            token_counter=lambda r: r.usage.total_tokens,
        )
        def agent_step(prompt: str) -> LLMResponse:
            return llm.invoke(prompt)

        # Access the guard for introspection:
        print(agent_step.guard.total_tokens)
        print(agent_step.guard.energy_history)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        guard = BoundaryGuard(
            token_budget=token_budget,
            lambda_=lambda_,
            window=window,
            on_violation=on_violation,
            on_budget_exhausted=on_budget_exhausted,
        )
        # Activate the guard immediately (no explicit with-block needed)
        guard._start_time = time.monotonic()
        guard._active = True

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)

            tokens = 0
            if token_counter is not None:
                try:
                    tokens = token_counter(result)
                except Exception:
                    tokens = 0

            guard.record_step(tokens_used=tokens)
            return result

        # Attach guard reference for external introspection
        wrapper.guard = guard  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ─── GrowthRatioGuard ───────────────────────────────────────────────────────


@dataclass
class GrowthRatioConfig:
    """Configuration for the GrowthRatioGuard.

    Attributes:
        token_budget: Maximum cumulative tokens before budget exhaustion.
        ratio_threshold: Growth ratio above which a turn counts as "escalating".
            A value of 2.0 means the monitor trips when a turn uses 2× the
            baseline (median of the first ``warmup_turns`` turns).
            When ``adaptive`` is True, this is auto-calibrated from warmup
            variance and this value serves as a fallback minimum.
        window: Consecutive escalating turns before circuit breaker trips.
        warmup_turns: Number of initial turns used to establish the baseline.
            The monitor does not evaluate growth during warmup.
        budget_gate: Minimum cumulative tokens before the monitor can trip.
            Prevents tripping on cheap tasks where even 2× growth is negligible.
        lambda_: Coupling constant for error weighting (passed to inner monitor).
        adaptive: If True, auto-calibrate ratio_threshold from warmup variance
            using Tukey's fence method: τ = 1 + k × (IQR / median).
            Eliminates per-domain manual tuning.
        adaptive_k: Sensitivity multiplier for Tukey's fence (default 2.0).
            Higher = more permissive. 1.5 is standard for outlier detection;
            2.0 is recommended for agent monitoring (conservative).
    """

    token_budget: int = 100_000
    ratio_threshold: float = 2.0
    window: int = 3
    warmup_turns: int = 3
    budget_gate: int = 8_000
    lambda_: float = 1.0
    adaptive: bool = False
    adaptive_k: float = 2.0


class GrowthRatioGuard:
    """Lyapunov guard with growth-ratio-based energy normalization.

    Instead of feeding raw token counts to the Lyapunov monitor, this guard
    normalizes each turn's tokens against a baseline (median of the first
    ``warmup_turns`` turns). The inner ``BoundaryGuard`` then sees a
    *growth ratio* as the energy signal, which eliminates false positives
    from the natural monotonic growth of multi-turn context windows.

    The guard also enforces a ``budget_gate``: even if the growth ratio
    exceeds the threshold, the circuit breaker won't trip until cumulative
    tokens exceed a minimum spend. This prevents killing cheap tasks where
    ratio spikes are harmless.

    Example::

        guard = GrowthRatioGuard(
            token_budget=100_000,
            ratio_threshold=2.0,  # trip when turn is 2× the baseline
            window=3,             # 3 consecutive escalating turns to trip
            budget_gate=8_000,    # don't trip until 8K tokens spent
        )

        with guard:
            for turn in agent_loop:
                result = llm.invoke(turn.prompt)
                guard.record_step(
                    tokens_used=result.usage.total_tokens,
                    errors=turn.retry_count,
                )
    """

    def __init__(
        self,
        token_budget: int = 100_000,
        ratio_threshold: float = 2.0,
        window: int = 3,
        warmup_turns: int = 3,
        budget_gate: int = 8_000,
        lambda_: float = 1.0,
        adaptive: bool = False,
        adaptive_k: float = 2.0,
        on_violation: Optional[Callable[[StabilityViolation], None]] = None,
        on_budget_exhausted: Optional[Callable[[BudgetExhausted], None]] = None,
    ) -> None:
        self._config = GrowthRatioConfig(
            token_budget=token_budget,
            ratio_threshold=ratio_threshold,
            window=window,
            warmup_turns=warmup_turns,
            budget_gate=budget_gate,
            lambda_=lambda_,
            adaptive=adaptive,
            adaptive_k=adaptive_k,
        )
        # The inner guard uses a very high window (999) because we handle
        # the windowed ratio check ourselves. The inner guard still enforces
        # the budget ceiling.
        self._inner = BoundaryGuard(
            token_budget=token_budget,
            lambda_=lambda_,
            window=999,  # disable inner window — we manage it
            on_violation=on_violation,
            on_budget_exhausted=on_budget_exhausted,
        )
        self._inner._start_time = time.monotonic()
        self._inner._active = True

        self._on_violation = on_violation

        # Per-turn raw token history for ratio computation
        self._turn_tokens: list[int] = []
        self._baseline: Optional[float] = None
        self._consecutive_escalating: int = 0

    # ── Context Manager Protocol ────────────────────────────────────────

    def __enter__(self) -> GrowthRatioGuard:
        self._inner.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        return self._inner.__exit__(exc_type, exc_val, exc_tb)

    # ── Primary API ─────────────────────────────────────────────────────

    def record_step(
        self,
        tokens_used: int,
        errors: int = 0,
        tool_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StabilityStatus:
        """Record metrics for a single turn and check growth-ratio stability.

        During warmup (first ``warmup_turns`` turns), tokens are recorded
        but no growth check is performed. After warmup, each turn's token
        usage is compared to the baseline. If the growth ratio exceeds
        ``ratio_threshold`` for ``window`` consecutive turns AND cumulative
        tokens exceed ``budget_gate``, ``StabilityViolation`` is raised.

        Args:
            tokens_used: Tokens consumed in this turn (instantaneous, not cumulative).
            errors: Number of transient errors in this turn.
            tool_name: Optional name of the tool called.
            metadata: Optional metadata for diagnostics.

        Returns:
            Current stability status.

        Raises:
            StabilityViolation: Growth ratio exceeded threshold for ``window``
                consecutive turns past the budget gate.
            BudgetExhausted: Cumulative tokens exceed the budget ceiling.
        """
        self._turn_tokens.append(tokens_used)

        # Always record in the inner guard (for budget tracking + telemetry)
        status = self._inner.record_step(
            tokens_used=tokens_used,
            errors=errors,
            tool_name=tool_name,
            metadata=metadata,
        )

        # Compute baseline after warmup period
        warmup = self._config.warmup_turns
        if len(self._turn_tokens) == warmup:
            sorted_tokens = sorted(self._turn_tokens[:warmup])
            n_warmup = len(sorted_tokens)
            self._baseline = float(sorted_tokens[n_warmup // 2])

            # Adaptive threshold: auto-calibrate τ from warmup variance
            # Uses Tukey's fence: τ = 1 + k × (IQR / median)
            # High-variance domains (telecom, retail) get a higher τ;
            # low-variance domains (airline) get a tighter τ.
            if self._config.adaptive and self._baseline > 0 and n_warmup >= 4:
                q1 = float(sorted_tokens[n_warmup // 4])
                q3 = float(sorted_tokens[3 * n_warmup // 4])
                iqr = q3 - q1
                adaptive_tau = 1.0 + self._config.adaptive_k * (iqr / self._baseline)
                # Clamp: minimum 1.5 (any lower is too trigger-happy),
                # maximum 5.0 (any higher defeats the purpose)
                adaptive_tau = max(1.5, min(5.0, adaptive_tau))
                self._config.ratio_threshold = adaptive_tau
                import logging
                logging.getLogger(__name__).info(
                    f"[state-harness] Adaptive threshold calibrated: "
                    f"τ={adaptive_tau:.2f} "
                    f"(median={self._baseline:.0f}, IQR={iqr:.0f}, "
                    f"k={self._config.adaptive_k})"
                )
            elif self._config.adaptive and n_warmup < 4:
                # Not enough warmup data for IQR — use conservative fallback
                self._config.ratio_threshold = max(
                    self._config.ratio_threshold, 2.5
                )

        # Skip ratio check during warmup or if baseline is too small
        if self._baseline is None or self._baseline <= 0:
            return status

        if len(self._turn_tokens) <= warmup:
            return status

        # Check growth ratio
        ratio = tokens_used / self._baseline
        cumulative = sum(self._turn_tokens)

        if ratio > self._config.ratio_threshold and cumulative > self._config.budget_gate:
            self._consecutive_escalating += 1
        else:
            self._consecutive_escalating = 0

        if self._consecutive_escalating >= self._config.window:
            exc = StabilityViolation(
                f"Growth ratio violation: {ratio:.1f}× baseline "
                f"for {self._consecutive_escalating} consecutive turns "
                f"(threshold: {self._config.ratio_threshold}×, "
                f"window: {self._config.window}). "
                f"Baseline: {self._baseline:.0f} tokens/turn. "
                f"Cumulative: {cumulative} tokens."
            )
            if self._on_violation is not None:
                self._on_violation(exc)
                return StabilityStatus.Tripped
            raise exc

        return status

    def reset_escalation(self) -> None:
        """Reset the consecutive escalating counter after a causal intervention.

        Called by the agent after applying RG compression to give the agent
        a fresh chance to stabilize with compressed context. This is the
        "feedback" step in the closed-loop Lyapunov controller (§6 of the
        fluid dynamics paper).
        """
        self._consecutive_escalating = 0

    def report_transient(self) -> StabilityStatus:
        """Report a transient failure."""
        return self._inner.report_transient()

    def report_permanent(self, reason: str) -> None:
        """Report a permanent failure."""
        self._inner.report_permanent(reason)

    def check(self) -> StabilityStatus:
        """Check current stability without recording new metrics."""
        return self._inner.check()

    # ── Introspection Properties ────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all recorded turns."""
        return sum(self._turn_tokens)

    @property
    def total_steps(self) -> int:
        """Total number of recorded turns."""
        return len(self._turn_tokens)

    @property
    def baseline(self) -> Optional[float]:
        """Baseline tokens per turn (median of first warmup turns)."""
        return self._baseline

    @property
    def current_ratio(self) -> Optional[float]:
        """Current growth ratio (latest turn / baseline)."""
        if self._baseline and self._baseline > 0 and self._turn_tokens:
            return self._turn_tokens[-1] / self._baseline
        return None

    @property
    def consecutive_escalating(self) -> int:
        """Number of consecutive turns exceeding the growth ratio threshold."""
        return self._consecutive_escalating

    @property
    def energy_history(self) -> list[float]:
        """Growth ratio trajectory (for diagnostics/visualization)."""
        if self._baseline is None or self._baseline <= 0:
            return [1.0] * len(self._turn_tokens)
        return [t / self._baseline for t in self._turn_tokens]

    @property
    def step_history(self) -> list[StepMetrics]:
        """Delegate to inner guard's step history."""
        return self._inner.step_history

    @property
    def is_stable(self) -> bool:
        """Whether the system is currently stable."""
        return self._consecutive_escalating < self._config.window

    @property
    def is_tripped(self) -> bool:
        """Whether the circuit breaker has been tripped."""
        return (
            self._consecutive_escalating >= self._config.window
            or self._inner.is_tripped
        )

    @property
    def is_frozen(self) -> bool:
        """Whether the state has been frozen due to budget exhaustion."""
        return self._inner.is_frozen

    @property
    def snapshot(self) -> Any:
        """Compile a frozen telemetry snapshot."""
        return self._inner.snapshot

