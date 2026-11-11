# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-11-11

### Changed
- **README figures corrected to match the published paper** (no code changes):
  - Multi-trial SWE-bench results now come from the evaluation records: A 45.9% ± 4.7, D 41.4% ± 4.1, E 47.7% ± 3.1; A−D −4.5pp, not significant (p = 0.28).
  - SWE-bench stability trips: 87 of 148 full-stack runs (59%), replacing ~38%.
  - τ³-bench: 24–34% of tasks (28% on average) have mixed pass/fail outcomes across trials; the unsourced ±4.6% and ±12pp noise figures are replaced with measured re-run spreads.
  - Zero false positives now refers to the 80 local-model harness runs. GPT-4o-mini and Claude Haiku 4.5 are listed as smoke-tested only. On SWE-bench, 10 of the 25 tasks full-stack monitoring stopped had been resolved by the baseline.

### Fixed
- Release workflow: install Python 3.10, 3.11 and 3.12 on every wheel-build runner, so the Windows wheels build again.
- `state_harness.__version__` now matches the package version (the 0.3.0 wheels reported 0.2.0).

## [0.3.0] - 2026-06-01

### Added
- **SafeGraph**: First-class LangGraph integration — wraps a compiled graph with automatic Lyapunov monitoring. Extracts token counts from `AIMessage.usage_metadata`. Supports both `invoke()` and `stream()`.
- **CrewAICallback**: Drop-in callback adapter for CrewAI crews. Hooks into `step_callback` and `task_callback` for automatic monitoring.
- **CLI tool**: `state-harness check` for live monitoring and `state-harness report` for post-mortem analysis from JSON telemetry files.
- **Adaptive ratio threshold**: Auto-calibrates growth ratio threshold from warmup variance using Tukey's fence method (`τ = 1 + k × IQR/median`), eliminating per-domain manual tuning.
- **Cross-model validation**: Verified zero false positives in 80 harness runs on 4 local model families (Llama 3.2:3B, Phi-4-Mini, Qwen3:4B, Gemma4:E4B).
- **Custom local-model benchmark battery**: 808 runs across 4 open-weight models on consumer hardware via Ollama. Revealed small-model self-sabotage pattern (+17.5pp naive-cap advantage).
- **MINT benchmark integration**: 1,136 runs across 4 MINT subsets (GSM8K, MATH, HumanEval, MBPP) with Gemini 2.5 Flash and Qwen3:4B.
- **OpenTelemetry export**: `FailureReport.to_otel_attributes()` for structured observability integration.
- **CSV export**: `FailureReport.to_csv_row()` and `csv_header()` for batch analysis.
- **"Who should / shouldn't use this"** section in README.

### Changed
- Expanded benchmark scope from 3 → 4 benchmarks (added custom local-model battery), total runs from 2,367 → 3,175.
- Updated self-sabotage statistics from median (+12.5pp) to mean (+17.5pp) after correcting for data corruption.
- Research paper updated from "three benchmarks" to "four benchmarks" throughout.

## [0.2.0] - 2026-05-31

### Added
- **GrowthRatioGuard**: Self-calibrating circuit breaker that normalizes token usage against a warmup baseline. Recommended over raw `BoundaryGuard` for most use cases.
- **FailureReport**: Zero-cost failure diagnostics — classifies failure patterns (spiral, retry storm, policy drift, early explosion, budget exhaustion), provides evidence, and suggests specific fixes. No LLM calls required.
- **FailurePattern enum**: Structured failure classification with confidence scores.
- **Suggestion dataclass**: Actionable fix recommendations with severity levels.
- **Framework adapters**: `LangGraphMiddleware` and `VanillaHook` for drop-in integration.
- **Benchmark results**: τ³-bench airline (58% pass, 9% savings), SWE-bench Verified (49.5% savings, 68.8% precision), MINT (0.8% savings, zero trips).
- **Research paper**: Published empirical validation across three benchmarks.

### Changed
- Bumped Cargo.toml version to match pyproject.toml.
- Standardized all repository URLs to canonical GitHub location.

## [0.1.0] - 2026-05-20

### Added
- **LyapunovMonitor** (Rust): Discrete-time energy tracker with circuit breaker semantics. Tracks V(k) = S(k) + λθ(k) and trips when ΔV ≥ 0 for W consecutive steps.
- **RGDecimator** (Rust): TF-IDF-based conversation history compression with structural keyword retention. First and last messages always preserved.
- **HolographicEngine** (Rust): VSA-based policy drift detection using 10,000-dimensional bipolar hypervectors. Constant-time cosine similarity checks.
- **BoundaryGuard** (Python): Context manager wrapping the Lyapunov monitor with Python ergonomics.
- **`@boundary_guard` decorator**: Function-level monitoring for individual agent steps.
- **MonitorGroup**: Manage multiple monitors for multi-agent orchestration.
- **Custom exceptions**: `StabilityViolation`, `BudgetExhausted`, `PermanentFailure`.
- **Type stubs** (`_core.pyi`): Full type coverage for IDE support.
