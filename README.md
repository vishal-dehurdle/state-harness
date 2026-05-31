# state-harness 🌀

[![Rust Core](https://img.shields.io/badge/core-Rust-orange?logo=rust)](src/)
[![Python SDK](https://img.shields.io/badge/sdk-Python-blue?logo=python)](python/)
[![License: Split BSL/Apache](https://img.shields.io/badge/license-BSL%201.1%20%2F%20Apache%202.0-blueviolet)](LICENSE.md)

**Runtime safety layer for LLM agents.** Detects runaway token spirals, kills doomed tasks early, and tells you exactly why they failed — before they burn your budget.

```python
from state_harness import GrowthRatioGuard, FailureReport

guard = GrowthRatioGuard(token_budget=50_000)

with guard:
    for turn in agent_loop:
        result = llm.invoke(turn.prompt)
        guard.record_step(tokens_used=result.usage.total_tokens)

# What went wrong? (zero-cost, no LLM calls)
report = FailureReport.from_guard(guard)
print(report)
```

```
⚠️  STABILITY TRIPPED at turn 12

Pattern: Context Accumulation Spiral (confidence: 92%)
  • Last 5 turns all exceeded 1.5× baseline (4/4 were accelerating).
  • Peak growth ratio: 5.2× baseline.
  • Without intervention, projected cost was $0.0396 (actual: $0.0039).

Energy: ▁▁▁▁▁▂▂▃▄▆█
  Baseline: 1050 tokens/turn
  Peak ratio: 5.2× baseline

Cost: $0.0039 (saved ~$0.0357 by tripping early)

Suggested actions:
  🔴 1. Enable RG history compression in your agent loop.
     → Compressing older messages reduces prompt tokens by 40-60%.
  🟡 2. Lower the growth ratio threshold to 1.8×.
     → A lower threshold would have caught it earlier.
  🟢 3. Add a sliding-window context strategy.
     → Send only the last N messages plus a summary of earlier ones.
```

---

## Why this exists

Every team running LLM agents in production has experienced this: an agent gets stuck in a loop, token usage spirals, and you find a $15 charge for a single failed request the next morning.

Existing solutions are either **too simple** (hard budget caps that kill tasks indiscriminately) or **too complex** (platforms requiring dashboards, infrastructure, and vendor lock-in).

State-harness is a **library**, not a platform. `pip install` and go. It uses [Lyapunov stability theory](https://en.wikipedia.org/wiki/Lyapunov_stability) to detect runaway behavior *before* it becomes expensive — and when it does trip, it tells you exactly what went wrong and how to fix it.

### What it catches

| Pattern | Signal | Example |
|:---|:---|:---|
| **Context Spiral** | Token growth accelerating beyond baseline | Agent replaying full history each turn |
| **Retry Storm** | Low-variance repeated calls | Tool failing, agent retrying identically |
| **Policy Drift** | VSA similarity score dropping | Agent going off-topic mid-conversation |
| **Early Explosion** | Token spike in first 3 turns | Oversized system prompt or tool response |
| **Budget Exhaustion** | Cumulative spend hits ceiling | Complex task, not necessarily broken |

---

## Installation

```bash
pip install state-harness
```

Requires Python ≥ 3.10. Pre-built wheels are available for Linux, macOS, and Windows (x86_64 and ARM64). No Rust toolchain needed.

### From source (for development)

```bash
git clone https://github.com/vishal-dehurdle/state-harness.git
cd state-harness

python -m venv .venv && source .venv/bin/activate

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

pip install maturin
maturin develop --release

# Run tests
pip install pytest
pytest tests/
```

---

## Quickstart

### Basic: GrowthRatioGuard (recommended)

The `GrowthRatioGuard` normalizes token usage against a baseline, so it only trips on *disproportionate* growth — not the natural growth of multi-turn context windows.

```python
from state_harness import GrowthRatioGuard, StabilityViolation

guard = GrowthRatioGuard(
    token_budget=100_000,     # hard ceiling
    ratio_threshold=2.0,      # trip when turn is 2× the baseline
    window=3,                 # 3 consecutive escalating turns to trip
    budget_gate=8_000,        # don't trip until 8K tokens spent
)

with guard:
    for turn in agent_loop:
        try:
            result = llm.invoke(turn.prompt)
            guard.record_step(
                tokens_used=result.usage.total_tokens,
                errors=0,
            )
        except StabilityViolation as e:
            print(f"Agent killed: {e}")
            break

print(f"Total cost: {guard.total_tokens} tokens")
print(f"Baseline: {guard.baseline} tokens/turn")
print(f"Peak ratio: {guard.current_ratio}×")
```

### Failure Diagnostics

After any execution (tripped or not), get a structured failure report:

```python
from state_harness import FailureReport

report = FailureReport.from_guard(guard, model="gemini-2.5-flash")

# Human-readable terminal output
print(report)

# Structured dict for logging / dashboards
import json
print(json.dumps(report.to_dict(), indent=2))
```

The report classifies the failure pattern, provides evidence, estimates cost impact, and suggests specific fixes — all without any LLM calls.

### Classic: BoundaryGuard

For lower-level control using raw token counts (no normalization):

```python
from state_harness import BoundaryGuard

with BoundaryGuard(token_budget=100_000, lambda_=1.0, window=5) as guard:
    for turn in agent_loop:
        result = llm.invoke(turn.prompt)
        guard.record_step(
            tokens_used=result.usage.total_tokens,
            errors=0,
            tool_name="search",
        )
```

### Decorator: `@boundary_guard`

```python
from state_harness import boundary_guard

@boundary_guard(
    token_budget=50_000,
    token_counter=lambda r: r.usage.total_tokens,
)
def agent_step(prompt: str):
    return llm.invoke(prompt)
```

---

## Framework Integration

### LangGraph

```python
from state_harness import BoundaryGuard
from state_harness.adapters import LangGraphMiddleware

guard = BoundaryGuard(token_budget=150_000)
middleware = LangGraphMiddleware(guard)

@middleware.wrap_tool
def search_database(query: str):
    return db.search(query)

with guard:
    result = agent.invoke({"messages": [...]})
```

### Vanilla Python Hooks

```python
from state_harness import BoundaryGuard
from state_harness.adapters import VanillaHook

guard = BoundaryGuard(token_budget=50_000)
hook = VanillaHook(guard)

with guard:
    for step in agent_loop:
        hook.before_call(tool_name="search")
        result = execute_tool(step)
        hook.after_call(tokens_used=result.tokens)
```

---

## Architecture

State-harness combines three physics-inspired mechanisms, implemented in Rust for microsecond-speed enforcement:

```
Agent Loop
    │
    ▼
┌─────────────────────────────────────────┐
│  GrowthRatioGuard (Python SDK)          │
│  ├── Normalizes tokens → growth ratio   │
│  ├── Warmup baseline (first N turns)    │
│  └── Budget gate (min spend before trip)│
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────────┐
│Lyapunov│ │  RG    │ │Holographic │
│Monitor │ │Decim.  │ │  Engine    │
│        │ │        │ │   (VSA)    │
│V(k)=S+λθ│ │TF-IDF │ │ Drift     │
│ΔV ≥ 0? │ │Compress│ │ Detection │
└────────┘ └────────┘ └────────────┘
   Rust        Rust        Rust
```

| Component | Purpose | Speed |
|:---|:---|:---|
| **Lyapunov Monitor** | Tracks energy derivative ΔV(k). Trips when ΔV ≥ 0 for W consecutive steps. | ~1μs/step |
| **RG Decimator** | Compresses conversation history using TF-IDF scoring. Retains structurally important messages. | ~100μs/compress |
| **Holographic Engine** | VSA-based policy drift detection. Binds domain invariants to high-dimensional vectors. | ~10μs/check |

---

## Benchmarks

Evaluated across three complementary benchmarks spanning customer service, software engineering, and multi-turn reasoning. Full methodology and data in the [research paper](https://vishalvermalabs.com/papers/empirical-lyapunov-stability-agent-failure).

### Summary: Token savings scale with loop length

| Benchmark | Avg Turns | Token Savings | Trip Rate | Pass-Rate Impact |
|:---|:---|---:|---:|:---|
| **MINT** (reasoning + coding) | 1–5 | 0.8% | 0% | None (29.2% → 28.5%) |
| **τ³-bench** (customer service) | 3–10 | 9% | ~5% | None (58% preserved) |
| **SWE-bench Verified** (coding) | 10–50 | 49.5% | 43% | 10/15 resolved preserved |

**Key finding:** The monitor is non-invasive on short-loop agents (0–9% savings, zero false trips) and highly effective on long-loop agents where token spirals concentrate (49.5% savings, 68.8% precision).

### SWE-bench Verified (central result)

37 Django instances from SWE-bench Verified. Agent: moatless-tools SearchTree with 50-node budget. Model: Gemini 2.5 Flash.

| Metric | Baseline | State-Harness (τ=3.0, W=5) |
|:---|---:|---:|
| Total tokens | 19,838,665 | 10,011,520 |
| **Token savings** | — | **49.5%** |
| Resolved | 15 / 37 | 10 / 37 |
| **Precision** | — | **68.8%** |

The worst spirals — failed tasks consuming 1.5–1.6M tokens — are terminated at 74–83% savings per task. The guard trips at a median of node 23 (~46% through the budget), early enough for substantial savings but late enough for calibration.

**Threshold sensitivity (Pareto frontier):**

| τ | W | Trips | Token Savings | Resolved Preserved | Precision |
|:---|:---|---:|---:|---:|---:|
| 1.5 | 3 | 32 | 83.1% | 3 / 15 | 62.5% |
| 2.0 | 5 | 27 | 69.1% | 6 / 15 | 66.7% |
| **3.0** | **5** | **16** | **49.5%** | **10 / 15** | **68.8%** |
| 5.0 | 3 | 9 | 24.0% | 12 / 15 | 66.7% |

### τ³-bench Airline

50 tasks × 3 configurations. Agent handles airline reservations via tool calls.

| Config | Pass Rate | Token Savings | Notes |
|:---|---:|---:|:---|
| Baseline (no guard) | 58% | — | Full agent loop, no monitoring |
| Naive Cap (100K) | 58% | 0% | No airline task exceeds cap |
| **State-Harness** (τ=2.0) | **58%** | **9%** | Non-invasive: same pass rate, fewer tokens |

The naive cap achieves 0% savings because no airline task exceeds 100K tokens — confirming that hard budget caps provide no value when set above the task's natural consumption envelope.

### MINT (non-invasiveness validation)

284 tasks across GSM8K (48), MATH (100), HumanEval (45), MBPP (91). Agent uses up to 5 turns per task.

| Category | Baseline Success | Harness Success | Token Savings |
|:---|---:|---:|---:|
| reasoning/gsm8k | 91.7% | 91.7% | 3.0% |
| reasoning/math | 39.0% | 37.0% | −0.3% |
| coding/humaneval | 0.0% | 0.0% | 2.8% |
| coding/mbpp | 0.0% | 0.0% | 0.4% |
| **Total** | **29.2%** | **28.5%** | **0.8%** |

Zero stability violations recorded. The monitor correctly identifies short-loop tasks as stable and does not intervene.

### Reproducing the benchmarks

<details>
<summary>τ³-bench reproduction steps</summary>

```bash
# 1. Clone both repos
git clone https://github.com/vishal-dehurdle/state-harness.git
git clone https://github.com/sierra-research/tau-bench.git tau3-bench

# 2. Install state-harness
cd state-harness
python -m venv .venv && source .venv/bin/activate
pip install maturin && maturin develop --release

# 3. Install τ³-bench (with state-harness agent)
cd ../tau3-bench
uv sync
cp ../state-harness/tau3_integration/harness_agent.py src/tau2/agent/
cp ../state-harness/tau3_integration/naive_cap_agent.py src/tau2/agent/

# 4. Configure Vertex AI
export GOOGLE_CLOUD_PROJECT=your-project-id
export VERTEXAI_LOCATION=asia-south1  # or your preferred region

# 5. Run full benchmark
cd ../state-harness
./benchmarks/run_full_benchmark.sh
```

**Per-domain threshold tuning:**

| Domain | τ | Budget Gate | Rationale |
|:---|:---|:---|:---|
| Airline | 2.0 | 8,000 | Simple lookups; spirals are clear |
| **Retail** | **2.5** | **12,000** | Multi-item orders need more tokens per turn |
| Telecom | 2.0 | 8,000 | Sequential workflows; similar to airline |

</details>

See [benchmarks/](benchmarks/) for full setup, configs, and reproduction instructions for all three benchmarks.

### Future evaluations

- [ ] **Terminal-Bench** — Terminal-based agent tasks; tests command-line tool loops where spirals manifest as repeated failed commands
- [ ] **SWE-bench Pro** — Harder, contamination-resistant variant of SWE-bench
- [ ] **LiveCodeBench** — Freshly sampled coding problems with no training data overlap
- [ ] **Cross-model validation** — GPT-4o, Claude Sonnet 4, Llama 4 to validate model-agnosticity

### Planned features

- [ ] Adaptive threshold — Auto-calibrate τ from warmup dynamics instead of fixed per-domain defaults
- [ ] Causal intervention — Instead of killing spiraling tasks, redirect them (e.g., inject summary, reset context)
- [ ] Streaming support — Real-time monitoring for streaming/voice agents

---

## Configuration Guide

| Parameter | Default | Description |
|:---|:---|:---|
| `token_budget` | 100,000 | Hard ceiling on cumulative tokens |
| `ratio_threshold` | 2.0 | Growth ratio above which a turn counts as "escalating" (domain-tuned: airline=2.0, retail=2.5, telecom=2.0) |
| `window` | 3 | Consecutive escalating turns before circuit breaker trips |
| `warmup_turns` | 3 | Turns used to establish baseline (no monitoring during warmup) |
| `budget_gate` | 8,000 | Minimum cumulative tokens before the monitor can trip (retail: 12,000) |
| `lambda_` | 1.0 | Error weighting in the Lyapunov energy function |

**Environment variable overrides** (highest precedence, for threshold sweeps):

| Env Var | Description |
|:---|:---|
| `HARNESS_RATIO_THRESHOLD` | Override ratio_threshold (e.g., `2.5`) |
| `HARNESS_BUDGET_GATE` | Override budget_gate (e.g., `12000`) |

**Tuning tips:**
- **More aggressive** (catch spirals earlier): `ratio_threshold=1.8, window=2`
- **More conservative** (fewer false positives): `ratio_threshold=2.5, window=3`
- **High-value tasks**: Increase `budget_gate` to 20K+ to let expensive tasks run longer
- **Complex domains** (retail, multi-tool): Start with `ratio_threshold=2.5`

---

## Theoretical Foundations

State-harness applies control theory to LLM agent execution:

- **Lyapunov stability**: The energy function V(k) = S(k) + λθ(k) models token consumption as a dynamical system. When ΔV ≥ 0 for W consecutive steps, the system is provably unstable.
- **Renormalization Group (RG) theory**: Message compression is modeled as coarse-graining — eliminating high-frequency noise while preserving scale-invariant task objectives.
- **Vector Symbolic Architecture (VSA)**: Domain policies are bound to high-dimensional bipolar vectors (10,000-d, i8 space), enabling constant-time semantic drift detection outside the LLM context window.

---

## Research

This library implements the framework described in:

> **Empirical Lyapunov Stability: Growth-Ratio Energy Functions as Leading Indicators of Agent Task Failure**
> Vishal Verma, 2026
> [Read the full paper →](https://vishalvermalabs.com/papers/empirical-lyapunov-stability-agent-failure)

Key findings from the paper:
- Growth-ratio normalization achieves **49.5% token savings** on SWE-bench with **68.8% precision**
- The monitor is **non-invasive on short-loop agents** (0.8% savings on MINT, zero false trips)
- Token savings scale with loop length: 0.8% → 9% → 49.5% (MINT → τ³ → SWE-bench)
- The guard trips at a **median of node 23** (~46% through the budget), early enough for savings, late enough for calibration

Based on the theoretical framework from:
> **The Fluid Dynamics of Multi-Agent AI: Resolving d'Alembert's Paradox of Generative Workflows**
> Vishal Verma, 2026
> [Read →](https://vishalvermalabs.com/papers/fluid-dynamics-multi-agent-ai)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev environment setup, code style, and PR guidelines.

---

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md). Please do **not** open public issues for security reports.

---

## License

Split-core licensing:

| Component | License | Notes |
|:---|:---|:---|
| **Rust Core** (`src/`) | BSL 1.1 | Free for non-commercial + ARR < $1M. Converts to Apache 2.0 on May 26, 2030. |
| **Python SDK** (`python/`) | Apache 2.0 | Fully permissive. |

See [LICENSE.md](LICENSE.md) for full details.