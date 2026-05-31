# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Drop-in adapter hooks for popular agent frameworks.

Provides integration layers for LangGraph and vanilla Python agent loops,
enabling seamless instrumentation with state-harness stability monitoring.

Framework support:
- **LangGraph**: Middleware wrapping tool calls with stability tracking.
- **Vanilla Python**: Simple before/after hooks for custom agent loops.

Additional framework adapters (CrewAI, AutoGen) are planned for v2.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from state_harness._core import (
    BudgetExhausted,
    PermanentFailure,
    StabilityStatus,
    StabilityViolation,
)
from state_harness.sdk import BoundaryGuard


# ─── Protocols ──────────────────────────────────────────────────────────────


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for extracting token counts from tool/LLM results."""

    def __call__(self, result: Any) -> int: ...


# ─── Shared Data Structures ────────────────────────────────────────────────


@dataclass
class ToolCallContext:
    """Context information for an intercepted tool call.

    Captured by both VanillaHook and LangGraphMiddleware to build
    a structured execution trace for post-mortem analysis.

    Attributes:
        tool_name: Name of the tool or function called.
        arguments: Arguments passed to the tool.
        timestamp: Monotonic time when the call started.
        tokens_used: Tokens consumed by this call.
        errors: Number of errors/retries during this call.
        result: The return value of the tool call.
        duration_ms: Wall-clock duration of the call in milliseconds.
    """

    tool_name: str
    arguments: dict[str, Any]
    timestamp: float
    tokens_used: int = 0
    errors: int = 0
    result: Any = None
    duration_ms: float = 0.0


# ─── Vanilla Python Hook ───────────────────────────────────────────────────


class VanillaHook:
    """Simple callback-based hook for non-framework agent loops.

    Provides before/after call instrumentation that feeds metrics
    into a ``BoundaryGuard`` for Lyapunov stability monitoring.

    Use this when your agent loop is plain Python code without a
    framework like LangGraph or CrewAI.

    Example::

        guard = BoundaryGuard(token_budget=50_000)
        hook = VanillaHook(guard)

        with guard:
            for step in agent_loop:
                hook.before_call(tool_name="search")
                result = execute_tool(step)
                hook.after_call(
                    tokens_used=result.tokens,
                    errors=0,
                    result=result,
                )
    """

    def __init__(
        self,
        guard: BoundaryGuard,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self._guard = guard
        self._token_counter = token_counter
        self._current_context: Optional[ToolCallContext] = None
        self._call_history: list[ToolCallContext] = []

    def before_call(
        self,
        tool_name: str = "unknown",
        arguments: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record the start of a tool/LLM call.

        Args:
            tool_name: Name of the tool or function being called.
            arguments: Arguments passed to the tool.
        """
        self._current_context = ToolCallContext(
            tool_name=tool_name,
            arguments=arguments or {},
            timestamp=time.monotonic(),
        )

    def after_call(
        self,
        tokens_used: int = 0,
        errors: int = 0,
        result: Any = None,
    ) -> StabilityStatus:
        """Record the completion of a tool/LLM call and check stability.

        If ``tokens_used`` is 0 and a ``token_counter`` was provided at
        construction, it will be used to extract token count from ``result``.

        Args:
            tokens_used: Tokens consumed by this call.
            errors: Number of errors/retries during this call.
            result: The result of the tool call (used for token counting).

        Returns:
            Current stability status after recording.

        Raises:
            StabilityViolation: If stability threshold is breached.
            BudgetExhausted: If token budget is exceeded.
        """
        if tokens_used == 0 and result is not None and self._token_counter is not None:
            try:
                tokens_used = self._token_counter(result)
            except Exception:
                pass

        tool_name = "unknown"
        if self._current_context is not None:
            self._current_context.tokens_used = tokens_used
            self._current_context.errors = errors
            self._current_context.result = result
            self._current_context.duration_ms = (
                time.monotonic() - self._current_context.timestamp
            ) * 1000
            self._call_history.append(self._current_context)
            tool_name = self._current_context.tool_name
            self._current_context = None

        return self._guard.record_step(
            tokens_used=tokens_used,
            errors=errors,
            tool_name=tool_name,
        )

    @property
    def call_history(self) -> list[ToolCallContext]:
        """Full history of intercepted tool calls."""
        return list(self._call_history)


# ─── LangGraph Middleware ───────────────────────────────────────────────────


# Permanent error indicators for automatic failure classification.
_PERMANENT_ERROR_INDICATORS: list[str] = [
    "schema",
    "validation",
    "invalid",
    "not found",
    "permission",
    "forbidden",
    "unauthorized",
    "configuration",
    "missing",
    "constraint",
]


class LangGraphMiddleware:
    """Middleware for LangGraph agent graphs.

    Intercepts tool calls within a LangGraph execution graph,
    feeding token metrics into a Lyapunov stability monitor.

    Provides two integration patterns:

    1. **Tool wrapping** via ``wrap_tool()`` / ``instrument()``:
       Decorate individual tool functions with stability monitoring.

    2. **Model callback** via ``create_model_callback()``:
       Track model invocation tokens across the execution graph.

    Example::

        from langgraph.prebuilt import create_react_agent
        from state_harness import BoundaryGuard
        from state_harness.adapters import LangGraphMiddleware

        guard = BoundaryGuard(token_budget=100_000)
        middleware = LangGraphMiddleware(guard)

        @middleware.wrap_tool
        def search_database(query: str) -> str:
            return db.search(query)

        agent = create_react_agent(
            model,
            tools=[search_database],
        )

        with guard:
            result = agent.invoke({"messages": [...]})
    """

    def __init__(
        self,
        guard: BoundaryGuard,
        token_counter: Optional[TokenCounter] = None,
        on_tool_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        self._guard = guard
        self._token_counter = token_counter
        self._on_tool_error = on_tool_error
        self._call_log: list[ToolCallContext] = []

    def wrap_tool(self, func: Callable) -> Callable:
        """Decorator that wraps a tool function with stability monitoring.

        The wrapped function automatically records token usage
        and errors in the ``BoundaryGuard`` after each invocation.

        Tool errors are classified as transient or permanent:

        - **Permanent**: Exception message contains schema/validation/permission indicators.
        - **Transient**: All other exceptions (timeouts, rate limits, etc.)

        Args:
            func: The tool function to wrap.

        Returns:
            Instrumented tool function with the same signature.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_name = getattr(func, "__name__", "unknown_tool")
            start = time.monotonic()
            errors = 0
            result = None

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                errors = 1
                if self._on_tool_error is not None:
                    self._on_tool_error(tool_name, e)

                # Classify and report the error
                if _is_permanent_error(e):
                    self._guard.report_permanent(
                        f"Tool '{tool_name}' raised permanent error: {e}"
                    )
                else:
                    self._guard.report_transient()
                raise

            duration_ms = (time.monotonic() - start) * 1000
            tokens = self._extract_tokens(result)

            ctx = ToolCallContext(
                tool_name=tool_name,
                arguments={"args": args, "kwargs": kwargs},
                timestamp=start,
                tokens_used=tokens,
                errors=errors,
                result=result,
                duration_ms=duration_ms,
            )
            self._call_log.append(ctx)

            self._guard.record_step(
                tokens_used=tokens,
                errors=errors,
                tool_name=tool_name,
            )

            return result

        return wrapper

    def instrument(self, func: Callable) -> Callable:
        """Alias for ``wrap_tool()``.

        Instruments a tool function with stability monitoring.
        """
        return self.wrap_tool(func)

    def create_model_callback(self) -> Callable:
        """Create a callback for LangGraph model invocation tracking.

        Returns a callable that can be registered as a LangGraph callback
        to automatically track model token usage across the execution graph.

        Returns:
            Callback function for model invocation tracking.

        Example::

            middleware = LangGraphMiddleware(guard)
            callback = middleware.create_model_callback()

            # Pass as a callback to LangGraph model invocations
            model.invoke(prompt, config={"callbacks": [callback]})
        """

        def on_llm_end(response: Any, **kwargs: Any) -> None:
            tokens = self._extract_tokens(response)
            self._guard.record_step(
                tokens_used=tokens,
                tool_name="llm_invocation",
            )

        return on_llm_end

    @property
    def call_log(self) -> list[ToolCallContext]:
        """Full log of intercepted tool calls."""
        return list(self._call_log)

    def _extract_tokens(self, result: Any) -> int:
        """Extract token count from a result using configured or heuristic methods."""
        if self._token_counter is not None:
            try:
                return self._token_counter(result)
            except Exception:
                pass

        # Attempt common LangChain/OpenAI attribute patterns
        for attr_path in [
            "usage.total_tokens",
            "response_metadata.token_usage.total_tokens",
            "usage_metadata.total_tokens",
        ]:
            tokens = _resolve_attr(result, attr_path)
            if tokens is not None:
                try:
                    return int(tokens)
                except (TypeError, ValueError):
                    continue

        return 0


# ─── Utility Functions ──────────────────────────────────────────────────────


def _resolve_attr(obj: Any, path: str) -> Any:
    """Resolve a dotted attribute path on an object or dict."""
    current = obj
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _is_permanent_error(e: Exception) -> bool:
    """Classify whether an exception represents a permanent (unrecoverable) failure.

    Permanent failures trigger immediate circuit breaker trips.
    Transient failures are absorbed into the Lyapunov error weight θ(k).
    """
    error_str = str(e).lower()
    error_type = type(e).__name__.lower()
    return any(
        indicator in error_str or indicator in error_type
        for indicator in _PERMANENT_ERROR_INDICATORS
    )
