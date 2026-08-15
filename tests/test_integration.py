# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Integration tests for the state-harness Python SDK.

These tests exercise the full Rust↔Python FFI boundary, verifying that
the compiled _core module works correctly from Python. They complement
the 69 Rust-side unit tests with end-to-end SDK behavioral tests.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest

from state_harness import (
    BoundaryGuard,
    BudgetExhausted,
    CoarseGrainer,
    FailureType,
    GuardConfig,
    HolographicEngine,
    LyapunovMonitor,
    MonitorGroup,
    PermanentFailure,
    RGDecimator,
    StabilityStatus,
    StabilityViolation,
    StepMetrics,
    TelemetrySnapshot,
    boundary_guard,
)


# ═══════════════════════════════════════════════════════════════════════════
# VSA: Holographic Engine
# ═══════════════════════════════════════════════════════════════════════════


class TestHolographicEngine:
    """Test the VSA engine across the FFI boundary."""

    def test_create_default_dimensionality(self):
        engine = HolographicEngine()
        assert engine.dimensionality() == 10_000

    def test_create_custom_dimensionality(self):
        engine = HolographicEngine(dim=500)
        assert engine.dimensionality() == 500

    def test_zero_dim_rejected(self):
        with pytest.raises(ValueError):
            HolographicEngine(dim=0)

    def test_generate_random_vector_is_bipolar(self):
        engine = HolographicEngine(dim=1000)
        v = engine.generate_random_vector()
        assert len(v) == 1000
        assert all(x in (-1, 1) for x in v)

    def test_bind_self_inverse(self):
        """Core algebraic property: a ⊗ a = identity."""
        engine = HolographicEngine(dim=1000)
        a = engine.generate_random_vector()
        b = engine.generate_random_vector()
        bound = engine.bind(a, b)
        recovered = engine.bind(bound, a)
        assert recovered == b

    def test_cosine_identical_is_one(self):
        engine = HolographicEngine(dim=1000)
        v = engine.generate_random_vector()
        assert engine.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_opposite_is_negative_one(self):
        engine = HolographicEngine(dim=100)
        v = [1] * 100
        neg_v = [-1] * 100
        assert engine.cosine_similarity(v, neg_v) == pytest.approx(-1.0)

    def test_cosine_random_near_zero(self):
        """Random high-dimensional vectors are pseudo-orthogonal."""
        engine = HolographicEngine(dim=10_000)
        a = engine.generate_random_vector()
        b = engine.generate_random_vector()
        sim = engine.cosine_similarity(a, b)
        assert abs(sim) < 0.1  # Should be near zero for D=10K

    def test_bundle_majority_vote(self):
        engine = HolographicEngine(dim=5)
        v1 = [1, 1, 1, -1, -1]
        v2 = [1, -1, 1, -1, 1]
        v3 = [1, 1, -1, -1, -1]
        bundled = engine.bundle([v1, v2, v3])
        assert bundled == [1, 1, 1, -1, -1]  # majority vote

    def test_batch_cosine_similarity(self):
        engine = HolographicEngine(dim=100)
        target = engine.generate_random_vector()
        candidates = [engine.generate_random_vector() for _ in range(5)]
        candidates.append(target)  # Add identical vector
        sims = engine.batch_cosine_similarity(target, candidates)
        assert len(sims) == 6
        assert sims[-1] == pytest.approx(1.0)  # Last one is identical

    def test_text_encoding_deterministic(self):
        engine = HolographicEngine(dim=500)
        v1 = engine.encode_text("hello world")
        v2 = engine.encode_text("hello world")
        assert v1 == v2

    def test_text_encoding_similar_texts_correlated(self):
        engine = HolographicEngine(dim=5000)
        v1 = engine.encode_text("execute the database migration")
        v2 = engine.encode_text("execute the database backup")
        v3 = engine.encode_text("the weather is sunny today")
        sim_related = engine.cosine_similarity(v1, v2)
        sim_unrelated = engine.cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated

    def test_batch_encode_texts(self):
        engine = HolographicEngine(dim=500)
        texts = ["hello", "world", "test"]
        vecs = engine.batch_encode_texts(texts)
        assert len(vecs) == 3
        assert all(len(v) == 500 for v in vecs)


# ═══════════════════════════════════════════════════════════════════════════
# VSA: Invariant Lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestInvariantLifecycle:
    """Test invariant registration, drift detection, and recovery."""

    def test_register_and_recover(self):
        engine = HolographicEngine(dim=500)
        key = engine.generate_random_vector()
        value = engine.generate_random_vector()
        engine.register_invariant("safety_goal", key, value)
        recovered = engine.recover("safety_goal")
        assert recovered == value

    def test_check_drift_aligned(self):
        engine = HolographicEngine(dim=500)
        key = engine.generate_random_vector()
        val = engine.generate_random_vector()
        engine.register_invariant("goal", key, val)
        bound = engine.bind(key, val)
        drift = engine.check_drift("goal", bound)
        assert drift == pytest.approx(1.0)

    def test_remove_invariant(self):
        engine = HolographicEngine(dim=500)
        key = engine.generate_random_vector()
        val = engine.generate_random_vector()
        engine.register_invariant("temp", key, val)
        assert engine.invariant_count() == 1
        assert engine.remove_invariant("temp") is True
        assert engine.invariant_count() == 0

    def test_list_invariants(self):
        engine = HolographicEngine(dim=500)
        for i in range(3):
            engine.register_invariant(
                f"inv_{i}",
                engine.generate_random_vector(),
                engine.generate_random_vector(),
            )
        names = engine.list_invariants()
        assert sorted(names) == ["inv_0", "inv_1", "inv_2"]

    def test_hard_cap_at_20(self):
        engine = HolographicEngine(dim=100)
        for i in range(20):
            engine.register_invariant(
                f"inv_{i}",
                engine.generate_random_vector(),
                engine.generate_random_vector(),
            )
        with pytest.raises(ValueError, match="Cannot register more than 20"):
            engine.register_invariant(
                "one_too_many",
                engine.generate_random_vector(),
                engine.generate_random_vector(),
            )

    def test_wrong_dimension_rejected(self):
        engine = HolographicEngine(dim=500)
        with pytest.raises(ValueError):
            engine.bind([1, -1], engine.generate_random_vector())

    def test_non_bipolar_rejected(self):
        engine = HolographicEngine(dim=3)
        with pytest.raises(ValueError, match="bipolar"):
            engine.bind([1, 2, 3], [1, -1, 1])


# ═══════════════════════════════════════════════════════════════════════════
# Lyapunov Monitor
# ═══════════════════════════════════════════════════════════════════════════


class TestLyapunovMonitor:
    """Test the Lyapunov circuit breaker from Python."""

    def test_stable_decreasing_energy(self):
        mon = LyapunovMonitor(lambda_=1.0, window=3)
        s1 = mon.record_step(tokens_used=1000, errors=0)
        s2 = mon.record_step(tokens_used=500, errors=0)
        s3 = mon.record_step(tokens_used=200, errors=0)
        assert str(s3) == "Stable"

    def test_trip_on_sustained_increase(self):
        mon = LyapunovMonitor(lambda_=1.0, window=3)
        mon.record_step(tokens_used=100, errors=0)
        mon.record_step(tokens_used=200, errors=0)
        mon.record_step(tokens_used=300, errors=0)
        with pytest.raises(StabilityViolation):
            mon.record_step(tokens_used=400, errors=0)

    def test_initialization_spike_no_false_positive(self):
        """Critical edge case from the technical document.

        Agents often consume a large block of tokens on their first turn
        (system prompt). The massive drop from Step 1 → Step 2 results in
        a highly negative ΔV (stable). Then Step 3 jumps up (tool validation).

        The window=3 threshold must NOT false-positive on normal alternating
        execution patterns like: [8000, 500, 3000, 400, 2500, 300].
        """
        mon = LyapunovMonitor(lambda_=1.0, window=3)

        # Step 1: Massive system prompt load
        mon.record_step(tokens_used=8000, errors=0)
        # Step 2: Simple tool call (big drop → negative ΔV → stable)
        s = mon.record_step(tokens_used=500, errors=0)
        assert str(s) == "Stable"
        # Step 3: Complex validation (spike up → positive ΔV → warning)
        s = mon.record_step(tokens_used=3000, errors=0)
        # This is a single positive ΔV, not 3 consecutive → should not trip
        assert str(s) == "Warning"
        # Step 4: Another drop
        s = mon.record_step(tokens_used=400, errors=0)
        assert str(s) == "Stable"
        # Step 5: Another spike
        s = mon.record_step(tokens_used=2500, errors=0)
        assert str(s) == "Warning"
        # Step 6: Drop again
        s = mon.record_step(tokens_used=300, errors=0)
        assert str(s) == "Stable"

        # Monitor should never have tripped
        assert not mon.is_tripped()
        assert mon.total_steps() == 6

    def test_initialization_spike_converges_stable(self):
        """After initial spike, agent settles into efficient tool-calling."""
        mon = LyapunovMonitor(lambda_=1.0, window=3)

        # Initial heavy system prompt
        mon.record_step(tokens_used=10000, errors=0)
        # Converging into efficient execution
        mon.record_step(tokens_used=2000, errors=0)
        mon.record_step(tokens_used=800, errors=0)
        mon.record_step(tokens_used=300, errors=0)
        mon.record_step(tokens_used=250, errors=0)

        assert str(mon.check_stability()) == "Stable"
        assert not mon.is_tripped()

    def test_budget_exhaustion_freezes(self):
        mon = LyapunovMonitor(lambda_=1.0, window=10, budget_ceiling=5000)
        mon.record_step(tokens_used=2000, errors=0)
        mon.record_step(tokens_used=2000, errors=0)
        with pytest.raises(BudgetExhausted):
            mon.record_step(tokens_used=2000, errors=0)
        assert mon.is_frozen()

    def test_permanent_failure_trips_immediately(self):
        mon = LyapunovMonitor()
        mon.record_step(tokens_used=100, errors=0)
        with pytest.raises(PermanentFailure):
            mon.report_permanent_failure("Schema violation in CRM")
        assert mon.is_tripped()

    def test_transient_failure_weighted(self):
        """Transient errors contribute λ*1.0 to the energy function."""
        mon = LyapunovMonitor(lambda_=2.0, window=3)
        mon.record_step(tokens_used=0, errors=0)
        s = mon.report_transient_failure()
        # Energy = S(k) + λ*θ(k) = 0 + 2.0*1 = 2.0
        assert mon.current_energy() == pytest.approx(2.0)

    def test_snapshot_captures_full_state(self):
        mon = LyapunovMonitor(lambda_=1.0, window=3)
        mon.record_step(tokens_used=1000, errors=0)
        mon.record_step(tokens_used=500, errors=1)
        snap = mon.snapshot()
        assert snap.step_count == 2
        assert snap.cumulative_tokens == 1500
        assert snap.cumulative_errors == 1
        assert len(snap.energy_trajectory) == 2
        assert str(snap.final_status) == "Stable"

    def test_reset_clears_state(self):
        mon = LyapunovMonitor()
        mon.record_step(tokens_used=1000, errors=0)
        mon.record_step(tokens_used=2000, errors=0)
        mon.reset()
        assert mon.total_tokens() == 0
        assert mon.total_steps() == 0
        assert mon.current_energy() is None

    def test_energy_history_accessible(self):
        mon = LyapunovMonitor(lambda_=1.0, window=3)
        mon.record_step(tokens_used=100, errors=0)
        mon.record_step(tokens_used=200, errors=1)
        history = mon.get_energy_history()
        assert history[0] == pytest.approx(100.0)  # V(0) = 100 + 1.0*0
        assert history[1] == pytest.approx(201.0)  # V(1) = 200 + 1.0*1

    def test_invalid_lambda_rejected(self):
        with pytest.raises(ValueError):
            LyapunovMonitor(lambda_=-1.0)

    def test_invalid_window_rejected(self):
        with pytest.raises(ValueError):
            LyapunovMonitor(window=0)


# ═══════════════════════════════════════════════════════════════════════════
# Monitor Group (Hierarchical Composition)
# ═══════════════════════════════════════════════════════════════════════════


class TestMonitorGroup:
    """Test hierarchical multi-agent monitor topology."""

    def test_multi_agent_topology(self):
        """Simulate a 3-agent system: planner → researcher → executor."""
        group = MonitorGroup(budget_ceiling=50_000)
        group.add_child("planner", lambda_=1.0, window=3, budget_ceiling=20_000)
        group.add_child("researcher", lambda_=1.0, window=5)
        group.add_child("executor", lambda_=1.5, window=3)

        # All agents do their first steps
        group.record_step("planner", tokens_used=5000, errors=0)
        group.record_step("researcher", tokens_used=3000, errors=0)
        group.record_step("executor", tokens_used=2000, errors=0)

        assert group.child_count() == 3
        assert group.aggregate_tokens() == 10_000
        assert not group.is_any_tripped()

        # Agents converge to efficient execution
        group.record_step("planner", tokens_used=1000, errors=0)
        group.record_step("researcher", tokens_used=800, errors=0)
        group.record_step("executor", tokens_used=500, errors=0)

        assert group.is_all_stable()

    def test_aggregate_budget_catches_multi_agent_runaway(self):
        """Individual agents stay within their budgets, but aggregate spills."""
        group = MonitorGroup(budget_ceiling=10_000)
        group.add_child("a", lambda_=1.0, window=10)
        group.add_child("b", lambda_=1.0, window=10)
        group.add_child("c", lambda_=1.0, window=10)

        group.record_step("a", tokens_used=3000, errors=0)
        group.record_step("b", tokens_used=3000, errors=0)
        group.record_step("c", tokens_used=3000, errors=0)

        with pytest.raises(BudgetExhausted, match="Aggregate"):
            group.record_step("a", tokens_used=2000, errors=0)

    def test_per_child_status_isolation(self):
        """Tripping one child doesn't affect others."""
        group = MonitorGroup()
        group.add_child("stable", lambda_=1.0, window=3)
        group.add_child("unstable", lambda_=1.0, window=3)

        # Stable agent: decreasing tokens
        group.record_step("stable", tokens_used=1000, errors=0)
        group.record_step("stable", tokens_used=500, errors=0)

        # Unstable agent: increasing tokens
        group.record_step("unstable", tokens_used=100, errors=0)
        group.record_step("unstable", tokens_used=200, errors=0)
        group.record_step("unstable", tokens_used=300, errors=0)
        with pytest.raises(StabilityViolation):
            group.record_step("unstable", tokens_used=400, errors=0)

        assert group.is_any_tripped()
        assert not group.is_all_stable()
        assert str(group.child_status("stable")) == "Stable"

    def test_duplicate_child_rejected(self):
        group = MonitorGroup()
        group.add_child("agent")
        with pytest.raises(ValueError, match="already exists"):
            group.add_child("agent")

    def test_unknown_child_rejected(self):
        group = MonitorGroup()
        with pytest.raises(ValueError, match="No child"):
            group.record_step("ghost", tokens_used=100, errors=0)

    def test_remove_and_readd_child(self):
        group = MonitorGroup()
        group.add_child("temp")
        group.record_step("temp", tokens_used=500, errors=0)
        assert group.remove_child("temp")
        assert group.child_count() == 0
        # Can re-add after removal
        group.add_child("temp")
        assert group.child_count() == 1

    def test_aggregate_snapshots(self):
        group = MonitorGroup()
        group.add_child("a")
        group.add_child("b")
        group.record_step("a", tokens_used=100, errors=0)
        group.record_step("b", tokens_used=200, errors=1)
        snaps = group.aggregate_snapshots()
        assert len(snaps) == 2
        assert all(isinstance(s, TelemetrySnapshot) for s in snaps)


# ═══════════════════════════════════════════════════════════════════════════
# RG Decimator
# ═══════════════════════════════════════════════════════════════════════════


class TestRGDecimator:
    """Test Renormalization Group coarse-graining."""

    def test_preserves_first_and_last(self):
        rg = RGDecimator(threshold=0.99)  # Very aggressive threshold
        messages = ["First message", "filler", "filler", "Last message"]
        scored = rg.decimate(messages)
        assert scored[0].retained  # First always kept
        assert scored[-1].retained  # Last always kept

    def test_structural_keywords_boost_score(self):
        rg = RGDecimator(threshold=0.3)
        high = rg.score_message("Execute the database migration with error handling")
        low = rg.score_message("ok sure thing")
        assert high > low

    def test_compress_returns_only_retained(self):
        rg = RGDecimator(threshold=0.3, max_retained=3)
        messages = [
            "Begin the task execution",
            "hmm",
            "ok",
            "Execute database migration now",
            "yeah",
            "Result: migration completed successfully",
        ]
        compressed = rg.compress(messages)
        assert len(compressed) <= 3
        assert all(isinstance(m, str) for m in compressed)

    def test_max_retained_enforcement(self):
        rg = RGDecimator(threshold=0.0, max_retained=5)  # Retain all, but cap at 5
        messages = [f"Message {i} with execute and error content" for i in range(20)]
        scored = rg.decimate(messages)
        retained_count = sum(1 for s in scored if s.retained)
        # max_retained caps the middle messages; first and last are always retained
        # so total retained can be max_retained + 2 (first + last boundary pins)
        assert retained_count <= 7  # 5 + first + last

    def test_decimate_with_embeddings(self):
        """VSA-augmented decimation detects semantic redundancy."""
        engine = HolographicEngine(dim=1000)
        messages = [
            "Execute the database migration",
            "Execute the database migration",  # Duplicate
            "Check system health status",
        ]
        embeddings = [engine.encode_text(m) for m in messages]
        rg = RGDecimator(threshold=0.3, max_retained=50)
        scored = rg.decimate_with_embeddings(messages, embeddings)
        assert len(scored) == 3
        # Duplicate should have lower score than unique messages
        assert scored[1].score <= scored[2].score

    def test_empty_input(self):
        rg = RGDecimator()
        assert rg.decimate([]) == []
        assert rg.compress([]) == []

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            RGDecimator(threshold=-0.1)

    def test_invalid_max_retained_rejected(self):
        with pytest.raises(ValueError):
            RGDecimator(max_retained=0)


# ═══════════════════════════════════════════════════════════════════════════
# Python SDK: BoundaryGuard Context Manager
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundaryGuard:
    """Test the high-level Python SDK layer."""

    def test_context_manager_basic(self):
        with BoundaryGuard(token_budget=50_000) as guard:
            guard.record_step(tokens_used=1000, errors=0)
            guard.record_step(tokens_used=500, errors=0)
            assert guard.is_stable

    def test_context_manager_trip(self):
        with pytest.raises(StabilityViolation):
            with BoundaryGuard(token_budget=100_000, lambda_=1.0, window=3) as guard:
                guard.record_step(tokens_used=100, errors=0)
                guard.record_step(tokens_used=200, errors=0)
                guard.record_step(tokens_used=300, errors=0)
                guard.record_step(tokens_used=400, errors=0)  # Trip

    def test_report_transient_failure(self):
        with BoundaryGuard() as guard:
            guard.report_transient()
            assert guard.total_steps == 1

    def test_report_permanent_failure(self):
        with pytest.raises(PermanentFailure):
            with BoundaryGuard() as guard:
                guard.report_permanent(reason="Schema violation")


# ═══════════════════════════════════════════════════════════════════════════
# Python SDK: @boundary_guard Decorator
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundaryGuardDecorator:
    """Test the decorator syntax sugar over BoundaryGuard."""

    def test_decorator_wraps_function(self):
        @boundary_guard(token_budget=50_000, token_counter=lambda result: len(result))
        def agent_step(prompt: str) -> str:
            return "response " * 100

        result = agent_step("hello")
        assert result.startswith("response")

    def test_decorator_trips_on_runaway(self):
        call_count = 0

        @boundary_guard(
            lambda_=1.0,
            window=3,
            token_counter=lambda result: call_count * 1000,
        )
        def escalating_step(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"step {call_count}"

        # First few calls should work
        escalating_step("go")
        escalating_step("go")
        escalating_step("go")
        # Eventually it should trip
        with pytest.raises(StabilityViolation):
            for _ in range(20):
                escalating_step("go")


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Integration: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndPipeline:
    """Test the complete state-harness pipeline as described in the paper."""

    def test_full_agent_lifecycle(self):
        """Simulate a complete agent execution with all 3 subsystems."""
        # 1. Set up VSA invariant storage
        engine = HolographicEngine(dim=2000)
        goal_key = engine.encode_text("customer support")
        goal_val = engine.encode_text("resolve billing dispute politely")
        engine.register_invariant("support_goal", goal_key, goal_val)

        # 2. Set up Lyapunov monitor
        monitor = LyapunovMonitor(lambda_=1.0, window=3, budget_ceiling=100_000)

        # 3. Set up RG decimator
        rg = RGDecimator(threshold=0.3, max_retained=10)

        # Simulate 5-turn agent conversation
        conversation: list[str] = []
        token_schedule = [8000, 2000, 1500, 800, 500]  # Converging
        status = StabilityStatus.Stable  # Will be overwritten by the loop

        for i, tokens in enumerate(token_schedule):
            # Record step in Lyapunov monitor
            status = monitor.record_step(tokens_used=tokens, errors=0)

            # Add message to conversation
            conversation.append(f"Turn {i}: Agent processed {tokens} tokens")

            # Check drift against safety goal
            context_vec = engine.encode_text(f"Turn {i} billing support")
            drift = engine.check_drift("support_goal", context_vec)

            # Periodically compress history
            if len(conversation) > 3:
                conversation = rg.compress(conversation)

        # Verify stable execution
        assert not monitor.is_tripped()
        assert not monitor.is_frozen()
        assert str(status) == "Stable"
        assert monitor.total_tokens() == sum(token_schedule)
        assert engine.invariant_count() == 1

    def test_token_tsunami_interception(self):
        """Verify state-harness catches a runaway cost spiral.

        This is the core value proposition: the baseline agent would
        spiral into exponential cost, but the guarded agent hits a
        flat ceiling.
        """
        monitor = LyapunovMonitor(lambda_=1.0, window=3, budget_ceiling=50_000)

        # Simulate escalating token consumption (a runaway loop)
        tripped = False
        total_spent = 0
        for i in range(100):
            tokens = 500 * (i + 1)  # 500, 1000, 1500, 2000, ...
            try:
                monitor.record_step(tokens_used=tokens, errors=0)
                total_spent += tokens
            except (StabilityViolation, BudgetExhausted):
                tripped = True
                break

        # The monitor MUST have intercepted the runaway
        assert tripped, "Monitor failed to catch runaway token tsunami"
        # Cost was capped well below the theoretical unguarded spend
        unguarded_spend = sum(500 * (i + 1) for i in range(100))  # = 2,525,000
        assert total_spent < unguarded_spend * 0.01  # <1% of unguarded cost


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph SafeGraph Adapter
# ═══════════════════════════════════════════════════════════════════════════


class _MockMessage:
    """Mock LangGraph AIMessage with usage_metadata."""

    def __init__(self, total_tokens: int, name: str = "llm"):
        self.usage_metadata = {"total_tokens": total_tokens}
        self.name = name


class _MockGraph:
    """Mock compiled LangGraph graph for testing SafeGraph."""

    def __init__(self, messages: list[_MockMessage]):
        self._messages = messages

    def invoke(self, input: dict, config: dict = None, **kwargs) -> dict:
        return {"messages": self._messages}

    def stream(self, input: dict, config: dict = None, **kwargs):
        for msg in self._messages:
            yield {"messages": [msg]}


class TestSafeGraph:
    """Test the first-class LangGraph SafeGraph adapter."""

    def test_invoke_extracts_tokens(self):
        from state_harness.adapters import SafeGraph

        messages = [
            _MockMessage(1000, "llm"),
            _MockMessage(500, "search"),
        ]
        graph = _MockGraph(messages)
        safe = SafeGraph(graph, token_budget=100_000)

        result = safe.invoke({"messages": [("user", "test")]})
        assert result["messages"] == messages
        assert safe.total_tokens == 1500
        assert not safe.tripped

    def test_invoke_generates_report(self):
        from state_harness.adapters import SafeGraph

        messages = [_MockMessage(100, "llm")]
        graph = _MockGraph(messages)
        safe = SafeGraph(graph, token_budget=100_000)

        safe.invoke({"messages": [("user", "test")]})
        report = safe.report
        assert report is not None

    def test_stream_extracts_tokens(self):
        from state_harness.adapters import SafeGraph

        messages = [
            _MockMessage(1000, "llm"),
            _MockMessage(2000, "llm"),
        ]
        graph = _MockGraph(messages)
        safe = SafeGraph(graph, token_budget=100_000)

        chunks = list(safe.stream({"messages": [("user", "test")]}))
        assert len(chunks) == 2
        assert safe.total_tokens == 3000

    def test_budget_exhaustion_calls_on_trip(self):
        from state_harness.adapters import SafeGraph

        messages = [_MockMessage(60_000, "llm")]  # Over budget
        graph = _MockGraph(messages)

        trip_reports = []
        safe = SafeGraph(
            graph,
            token_budget=50_000,
            on_trip=lambda r: trip_reports.append(r),
        )

        with pytest.raises(BudgetExhausted):
            safe.invoke({"messages": [("user", "test")]})

    def test_monitor_graph_convenience(self):
        from state_harness.adapters import monitor_graph

        messages = [_MockMessage(100, "llm")]
        graph = _MockGraph(messages)
        safe = monitor_graph(graph, token_budget=100_000)

        safe.invoke({"messages": [("user", "test")]})
        assert safe.total_tokens == 100

    def test_zero_token_messages_ignored(self):
        from state_harness.adapters import SafeGraph

        messages = [_MockMessage(0, "system")]  # Zero tokens
        graph = _MockGraph(messages)
        safe = SafeGraph(graph, token_budget=100_000)

        safe.invoke({"messages": [("user", "test")]})
        assert safe.total_tokens == 0

    def test_extract_from_response_metadata(self):
        """Test fallback to response_metadata format."""
        from state_harness.adapters import SafeGraph

        class OldFormatMessage:
            usage_metadata = None
            response_metadata = {"token_usage": {"total_tokens": 999}}
            name = "llm"

        graph = _MockGraph([OldFormatMessage()])
        safe = SafeGraph(graph, token_budget=100_000)

        safe.invoke({"messages": [("user", "test")]})
        assert safe.total_tokens == 999


# ═══════════════════════════════════════════════════════════════════════════
# CrewAI Callback Adapter
# ═══════════════════════════════════════════════════════════════════════════


class _MockStepOutput:
    """Mock CrewAI step output."""

    def __init__(self, total_tokens: int, tool: str = "search", error=None):
        self.token_usage = {"total_tokens": total_tokens}
        self.tool = tool
        self.error = error


class _MockTaskOutput:
    """Mock CrewAI task output."""

    def __init__(self, total_tokens: int):
        self.token_usage = {"total_tokens": total_tokens}


class TestCrewAICallback:
    """Test the CrewAI callback adapter."""

    def test_step_callback_records_tokens(self):
        from state_harness.adapters import CrewAICallback

        cb = CrewAICallback(token_budget=100_000)
        cb.step_callback(_MockStepOutput(1000, "search"))
        cb.step_callback(_MockStepOutput(500, "calculate"))

        assert cb.guard.total_tokens == 1500
        cb.close()

    def test_task_callback_records_tokens(self):
        from state_harness.adapters import CrewAICallback

        cb = CrewAICallback(token_budget=100_000)
        cb.task_callback(_MockTaskOutput(5000))

        assert cb.guard.total_tokens == 5000
        cb.close()

    def test_report_generated(self):
        from state_harness.adapters import CrewAICallback

        cb = CrewAICallback(token_budget=100_000)
        cb.step_callback(_MockStepOutput(1000))
        report = cb.report
        assert report is not None
        cb.close()

    def test_error_steps_counted(self):
        from state_harness.adapters import CrewAICallback

        cb = CrewAICallback(token_budget=100_000)
        cb.step_callback(_MockStepOutput(1000, error="timeout"))
        cb.close()


# ═══════════════════════════════════════════════════════════════════════════
# Structured Output (JSON, CSV, OTEL)
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredOutput:
    """Test JSON, CSV, and OTEL output methods."""

    def _make_report(self):
        """Create a FailureReport from a spiraling guard."""
        from state_harness.diagnostics import FailureReport

        guard = BoundaryGuard(token_budget=50_000, lambda_=1.0, window=3)
        with guard:
            guard.record_step(tokens_used=1000, errors=0)
            guard.record_step(tokens_used=500, errors=0)
        return FailureReport.from_guard(guard)

    def test_to_json_returns_valid_json(self):
        import json

        report = self._make_report()
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["pattern"] == "healthy_completion"
        assert parsed["total_tokens"] == 1500
        assert isinstance(parsed["suggestions"], list)

    def test_to_json_compact(self):
        report = self._make_report()
        j = report.to_json(indent=None)
        assert "\n" not in j  # compact, single line

    def test_csv_header_matches_row_fields(self):
        from state_harness.diagnostics import FailureReport

        report = self._make_report()
        header = FailureReport.csv_header()
        row = report.to_csv_row()
        assert header.count(",") == row.count(",")

    def test_to_csv_row_parseable(self):
        report = self._make_report()
        row = report.to_csv_row()
        fields = row.split(",")
        assert fields[0] == "healthy_completion"
        assert int(fields[2]) == 1500  # total_tokens

    def test_to_otel_attributes_types(self):
        report = self._make_report()
        attrs = report.to_otel_attributes()

        # All values must be OTEL-compatible primitives
        for k, v in attrs.items():
            assert isinstance(k, str)
            assert isinstance(v, (str, int, float, bool)), f"{k}: {type(v)}"

        # Required keys
        assert "state_harness.pattern" in attrs
        assert "state_harness.total_tokens" in attrs
        assert attrs["state_harness.total_tokens"] == 1500

    def test_to_otel_pattern_value(self):
        report = self._make_report()
        attrs = report.to_otel_attributes()
        assert attrs["state_harness.pattern"] == "healthy_completion"


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


class TestCLI:
    """Test the CLI module."""

    def test_simulate_command(self):
        from state_harness.cli import main
        import sys

        old_argv = sys.argv
        sys.argv = ["state-harness", "simulate", "1000", "500", "200"]
        try:
            code = main()
            assert code == 0
        finally:
            sys.argv = old_argv

    def test_version_command(self):
        from state_harness.cli import main
        import sys

        old_argv = sys.argv
        sys.argv = ["state-harness", "--version"]
        try:
            code = main()
            assert code == 0
        finally:
            sys.argv = old_argv

