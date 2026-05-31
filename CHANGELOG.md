# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
