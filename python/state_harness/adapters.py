# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Drop-in adapter hooks for popular agent frameworks.

Provides integration layers for LangGraph, CrewAI, and vanilla Python agent loops,
enabling seamless instrumentation with state-harness stability monitoring.

Framework support:
- **LangGraph**: First-class ``SafeGraph`` wrapper (recommended) and tool-level ``LangGraphMiddleware``.
- **CrewAI**: ``CrewAICallback`` for step and task monitoring.
- **Vanilla Python**: Simple before/after hooks for custom agent loops.
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


# ─── First-Class LangGraph Integration ─────────────────────────────────────


class SafeGraph:
    """First-class LangGraph integration — wraps a compiled graph with stability monitoring.

    Instead of wrapping individual tools, ``SafeGraph`` wraps the entire compiled
    graph and automatically monitors every LLM invocation by extracting token
    counts from LangGraph's ``AIMessage.usage_metadata``.

    This is the recommended integration for LangGraph users — no need to
    understand the underlying guard API.

    Example::

        from langgraph.prebuilt import create_react_agent
        from state_harness.adapters import SafeGraph

        agent = create_react_agent(model, tools=[search, calculate])
        safe_agent = SafeGraph(agent, token_budget=100_000)

        result = safe_agent.invoke({"messages": [("user", "Fix the bug")]})

        # After execution:
        print(safe_agent.report)      # FailureReport (if tripped)
        print(safe_agent.total_tokens) # cumulative token usage

    Args:
        graph: A compiled LangGraph graph (e.g., from ``create_react_agent``).
        token_budget: Maximum cumulative tokens before budget exhaustion.
        ratio_threshold: Growth ratio threshold (default: 2.0).
        window: Consecutive escalating turns before trip (default: 3).
        budget_gate: Minimum cumulative tokens before monitoring activates (default: 8000).
        on_trip: Optional callback invoked when stability trips.
            Receives a ``FailureReport`` as argument.
    """

    def __init__(
        self,
        graph: Any,
        token_budget: int = 100_000,
        ratio_threshold: float = 2.0,
        window: int = 3,
        budget_gate: int = 8_000,
        on_trip: Optional[Callable] = None,
    ) -> None:
        from state_harness.sdk import GrowthRatioGuard
        from state_harness.diagnostics import FailureReport

        self._graph = graph
        self._guard = GrowthRatioGuard(
            token_budget=token_budget,
            ratio_threshold=ratio_threshold,
            window=window,
            budget_gate=budget_gate,
        )
        self._on_trip = on_trip
        self._report: Optional[FailureReport] = None
        self._FailureReport = FailureReport

    def invoke(
        self,
        input: Any,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the wrapped graph with stability monitoring.

        Automatically extracts token counts from each ``AIMessage`` in the
        output stream and feeds them to the growth-ratio guard.

        Args:
            input: Input to the graph (same as ``graph.invoke()``).
            config: Optional LangGraph config dict.
            **kwargs: Additional arguments passed to ``graph.invoke()``.

        Returns:
            The graph's output (same as ``graph.invoke()``).

        Raises:
            StabilityViolation: If growth ratio threshold is breached.
            BudgetExhausted: If token budget is exceeded.
        """
        config = config or {}

        with self._guard:
            try:
                result = self._graph.invoke(input, config=config, **kwargs)
                # Extract token usage from output messages
                self._extract_and_record(result)
                return result
            except (StabilityViolation, BudgetExhausted):
                self._report = self._FailureReport.from_guard(self._guard)
                if self._on_trip is not None:
                    self._on_trip(self._report)
                raise
            finally:
                # Always generate report for inspection
                if self._report is None:
                    self._report = self._FailureReport.from_guard(self._guard)

    def stream(
        self,
        input: Any,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Stream the wrapped graph with stability monitoring.

        Monitors each streamed chunk for token usage and stability.
        Yields the same chunks as ``graph.stream()``.

        Args:
            input: Input to the graph (same as ``graph.stream()``).
            config: Optional LangGraph config dict.
            **kwargs: Additional arguments passed to ``graph.stream()``.

        Yields:
            Same chunks as ``graph.stream()``.

        Raises:
            StabilityViolation: If growth ratio threshold is breached.
            BudgetExhausted: If token budget is exceeded.
        """
        config = config or {}

        with self._guard:
            try:
                for chunk in self._graph.stream(input, config=config, **kwargs):
                    self._extract_and_record(chunk)
                    yield chunk
            except (StabilityViolation, BudgetExhausted):
                self._report = self._FailureReport.from_guard(self._guard)
                if self._on_trip is not None:
                    self._on_trip(self._report)
                raise
            finally:
                if self._report is None:
                    self._report = self._FailureReport.from_guard(self._guard)

    def _extract_and_record(self, output: Any) -> None:
        """Extract token counts from LangGraph output and record them."""
        messages = self._get_messages(output)
        for msg in messages:
            tokens = self._extract_message_tokens(msg)
            if tokens > 0:
                tool_name = getattr(msg, "name", None) or "llm"
                self._guard.record_step(
                    tokens_used=tokens,
                    tool_name=tool_name,
                )

    @staticmethod
    def _get_messages(output: Any) -> list:
        """Extract messages from various LangGraph output formats."""
        if isinstance(output, dict):
            # Standard LangGraph output: {"messages": [...]}
            msgs = output.get("messages", [])
            if isinstance(msgs, list):
                return msgs
            return [msgs] if msgs else []
        if isinstance(output, list):
            return output
        return []

    @staticmethod
    def _extract_message_tokens(msg: Any) -> int:
        """Extract total token count from a LangGraph AIMessage.

        Supports:
        - ``msg.usage_metadata.total_tokens`` (LangChain BaseMessage)
        - ``msg.response_metadata.token_usage.total_tokens`` (older format)
        """
        # Try usage_metadata first (LangChain v0.2+)
        usage = getattr(msg, "usage_metadata", None)
        if usage is not None:
            total = None
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
            else:
                total = getattr(usage, "total_tokens", None)
            if total is not None:
                try:
                    return int(total)
                except (TypeError, ValueError):
                    pass

        # Fallback: response_metadata (older LangChain)
        resp_meta = getattr(msg, "response_metadata", None)
        if resp_meta is not None and isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage", {})
            if isinstance(token_usage, dict):
                total = token_usage.get("total_tokens")
                if total is not None:
                    try:
                        return int(total)
                    except (TypeError, ValueError):
                        pass

        return 0

    @property
    def report(self) -> Optional[Any]:
        """The failure report from the last execution, if available."""
        return self._report

    @property
    def guard(self) -> Any:
        """Direct access to the underlying GrowthRatioGuard for advanced use."""
        return self._guard

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all monitored invocations."""
        return self._guard.total_tokens

    @property
    def tripped(self) -> bool:
        """Whether the stability monitor tripped during the last execution."""
        return self._guard.is_tripped


def monitor_graph(
    graph: Any,
    token_budget: int = 100_000,
    ratio_threshold: float = 2.0,
    window: int = 3,
    budget_gate: int = 8_000,
    on_trip: Optional[Callable] = None,
) -> SafeGraph:
    """Wrap a compiled LangGraph graph with stability monitoring.

    This is the recommended entry point for LangGraph users.

    Example::

        from langgraph.prebuilt import create_react_agent
        from state_harness.adapters import monitor_graph

        agent = create_react_agent(model, tools=[search])
        safe = monitor_graph(agent, token_budget=100_000)

        result = safe.invoke({"messages": [("user", "Fix it")]})

    Args:
        graph: A compiled LangGraph graph.
        token_budget: Maximum cumulative tokens.
        ratio_threshold: Growth ratio threshold (default: 2.0).
        window: Consecutive escalating turns before trip (default: 3).
        budget_gate: Minimum spend before monitoring activates (default: 8000).
        on_trip: Optional callback when stability trips.

    Returns:
        A ``SafeGraph`` instance wrapping the graph.
    """
    return SafeGraph(
        graph=graph,
        token_budget=token_budget,
        ratio_threshold=ratio_threshold,
        window=window,
        budget_gate=budget_gate,
        on_trip=on_trip,
    )


# ─── CrewAI Integration ────────────────────────────────────────────────────


class CrewAICallback:
    """Callback for CrewAI task execution with stability monitoring.

    Hooks into CrewAI's callback system to monitor token usage
    across crew task executions.

    Example::

        from crewai import Agent, Task, Crew
        from state_harness.adapters import CrewAICallback

        callback = CrewAICallback(token_budget=200_000)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            step_callback=callback.step_callback,
            task_callback=callback.task_callback,
        )

        result = crew.kickoff()
        print(callback.report)

    Args:
        token_budget: Maximum cumulative tokens.
        ratio_threshold: Growth ratio threshold (default: 2.0).
        window: Consecutive escalating turns before trip (default: 3).
        budget_gate: Minimum spend before monitoring activates (default: 8000).
    """

    def __init__(
        self,
        token_budget: int = 200_000,
        ratio_threshold: float = 2.0,
        window: int = 3,
        budget_gate: int = 8_000,
    ) -> None:
        from state_harness.sdk import GrowthRatioGuard
        from state_harness.diagnostics import FailureReport

        self._guard = GrowthRatioGuard(
            token_budget=token_budget,
            ratio_threshold=ratio_threshold,
            window=window,
            budget_gate=budget_gate,
        )
        self._FailureReport = FailureReport
        self._guard.__enter__()

    def step_callback(self, step_output: Any) -> None:
        """CrewAI step callback — records token usage per agent step.

        Args:
            step_output: The CrewAI step output object.
        """
        tokens = 0

        # Extract tokens from CrewAI's step output
        token_usage = getattr(step_output, "token_usage", None)
        if token_usage is not None:
            if isinstance(token_usage, dict):
                tokens = token_usage.get("total_tokens", 0)
            else:
                tokens = getattr(token_usage, "total_tokens", 0)

        tool_name = getattr(step_output, "tool", "agent_step")
        errors = 1 if getattr(step_output, "error", None) else 0

        self._guard.record_step(
            tokens_used=int(tokens),
            errors=errors,
            tool_name=str(tool_name),
        )

    def task_callback(self, task_output: Any) -> None:
        """CrewAI task callback — records cumulative task token usage.

        Args:
            task_output: The CrewAI task output object.
        """
        token_usage = getattr(task_output, "token_usage", None)
        if token_usage is not None:
            tokens = 0
            if isinstance(token_usage, dict):
                tokens = token_usage.get("total_tokens", 0)
            else:
                tokens = getattr(token_usage, "total_tokens", 0)

            self._guard.record_step(
                tokens_used=int(tokens),
                tool_name="task_complete",
            )

    @property
    def report(self) -> Optional[Any]:
        """Generate a failure report from the current guard state."""
        return self._FailureReport.from_guard(self._guard)

    @property
    def guard(self) -> Any:
        """Direct access to the underlying GrowthRatioGuard."""
        return self._guard

    def close(self) -> None:
        """Close the guard context. Call this when the crew execution finishes."""
        self._guard.__exit__(None, None, None)

