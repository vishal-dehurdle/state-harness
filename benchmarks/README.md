# Benchmark Reproducibility Guide

This directory contains everything needed to reproduce the benchmark results
reported in our paper. All benchmarks compare three configurations:

| Config | Description |
|:---|:---|
| **Baseline** | Unmodified agent (no monitoring) |
| **Naive Cap** | Hard token/turn cap (common industry approach) |
| **Harness** | state-harness Lyapunov monitoring (our approach) |

## Benchmarks

### 1. τ³-bench (Airline Domain)
**What**: Customer-service dialog tasks — the agent handles airline reservations
via tool calls while interacting with a simulated user.

**Why**: Validates that state-harness is **non-invasive** on short-horizon tasks
where spirals are rare. We expect ≈0% pass-rate loss with ≈9% token savings.

```bash
cd benchmarks/tau3/
./run_5phase_airline.sh
```

### 2. SWE-bench (Software Engineering)
**What**: Real GitHub issues from open-source projects. The agent reads code,
writes patches, and runs tests — long-horizon tasks with complex state.

**Why**: This is where spirals **actually occur**. SWE-bench tasks often trigger
escalating tool-call loops (read → edit → test → fail → read → edit → ...).
State-harness should show clear token savings with equal or better resolve rate.

```bash
cd benchmarks/swe_bench/
./setup.sh        # One-time setup
./run_benchmark.sh --mode both --concurrency 3
```

### 3. MINT (Multi-turn Interaction)
**What**: Evaluates LLMs on multi-turn interaction tasks with tools and language
feedback. Covers reasoning (GSM8K, MATH), coding (HumanEval, MBPP), and more.

**Why**: MINT measures whether agents improve over multiple turns — exactly where
token spirals emerge. State-harness should detect and terminate spiral loops,
saving tokens without degrading multi-turn solve rates.

```bash
cd benchmarks/mint/
./setup_mint.sh       # One-time setup (clones mint-bench, installs deps)
./run_mint.sh --mode both   # Run baseline + harness
```

### 4. Custom Local Model Battery (Ollama)
**What**: Baseline/harness/naive-cap comparison across 4 open-weight models
(Llama 3.2:3B, Phi-4-Mini, Qwen3:4B, Gemma4:E4B) running locally via Ollama.
Covers both τ³-bench (airline domain) and MINT (reasoning + coding) tasks.

**Why**: Validates that state-harness works on **consumer hardware** with
**small models** that exhibit qualitatively different failure patterns than
cloud APIs — self-sabotage from naive caps is amplified (+17.5pp vs baseline).

**Prerequisites**:
- [Ollama](https://ollama.com) installed and running
- Models pulled: `ollama pull llama3.2:3b phi4-mini qwen3:4b gemma3:4b`

```bash
cd benchmarks/local_models/
./setup_local.sh                 # One-time setup (installs Python deps)
python run_local_benchmark.py    # Run full battery (≈6-8 hours)
python analyze_local_results.py  # Analyze results
```

### 5. AgentBench (Skipped — requires 16GB+ extra RAM)
**What**: Long-horizon agent tasks across OS, DB, WebShop, KnowledgeGraph.
Requires 16GB RAM for WebShop alone + 30GB Freebase download for KnowledgeGraph.
Not practical on machines with ≤16GB RAM.

---

## Prerequisites

### Software
- **Python 3.12+** (3.13 recommended)
- **Docker** or **OrbStack** (for SWE-bench)
- **uv** (recommended) or pip

### API Keys

| Key | Where to Get | Cost |
|:---|:---|:---|
| `VERTEXAI_PROJECT` | [Google Cloud Console](https://console.cloud.google.com) | Pay-as-you-go |
| `VERTEXAI_LOCATION` | Same | — |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON | — |
| `VOYAGE_API_KEY` | [Voyage AI](https://dash.voyageai.com/) | Free (200M tokens) |

### Environment Setup

```bash
# Create .env file in state-harness root
cat > .env << 'EOF'
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
VOYAGE_API_KEY=pa-your-voyage-key
EOF
```

---

## Directory Structure

```
benchmarks/
├── README.md                    ← This file
├── analyze_results.py           ← τ³-bench results analyzer
├── analyze_fullstack.py         ← Full-stack cross-benchmark analyzer
├── cross_model_smoke_test.py    ← Quick smoke test across model families
├── run_fullstack_benchmark.sh   ← Run all benchmarks end-to-end
├── tau3/
│   └── run_5phase_airline.sh    ← τ³-bench 5-phase airline runner
├── swe_bench/
│   ├── setup.sh                 ← One-time SWE-bench environment setup
│   ├── run_benchmark.sh         ← SWE-bench benchmark runner
│   ├── run_benchmark_bce.sh     ← Budget-ceiling-only variant
│   ├── run_benchmark_dbe.sh     ← Drift-based-early variant
│   ├── run_multi_trial.sh       ← Multi-trial runner for stdev measurement
│   ├── harness_loop.py          ← HarnessSearchTree integration
│   ├── docker_run.patch         ← Patches for moatless docker_run.py
│   └── flow_configs/
│       ├── swebench_baseline.json
│       └── swebench_harness.json
├── mint/
│   ├── setup_mint.sh            ← One-time MINT environment setup
│   ├── run_mint.sh              ← MINT benchmark runner
│   ├── run_mint_fullstack.sh    ← Full-stack MINT runner
│   ├── gemini_agent.py          ← Gemini agent for MINT tasks
│   ├── gemini_feedback_agent.py ← Gemini feedback agent
│   ├── mint_harness.py          ← Harness wrapper for MINT interactive loop
│   ├── run_harness_mint.py      ← Harness-mode MINT runner
│   └── configs/
│       ├── gemini_baseline_reasoning_gsm8k.json
│       ├── gemini_baseline_reasoning_math.json
│       ├── gemini_baseline_coding_humaneval.json
│       └── gemini_baseline_coding_mbpp.json
└── local_models/
    ├── setup_local.sh           ← One-time Ollama + deps setup
    ├── run_local_benchmark.py   ← Main local model benchmark runner
    ├── analyze_local_results.py ← Results analyzer for local runs
    ├── tasks.py                 ← Task definitions for local benchmarks
    ├── run_tau3_local.sh        ← τ³-bench runner for local models
    ├── run_mint_local.sh        ← MINT runner for local models
    └── run_mint_naive_cap.sh    ← MINT naive-cap runner for local models
```

---

## How It Works

### state-harness Integration with SWE-bench

SWE-bench uses [moatless-tools](https://github.com/aorwall/moatless-tools) to
run agents against real codebases inside Docker containers. The agent loop is
a `SearchTree` that iterates: select → expand → simulate → backpropagate.

We create a `HarnessSearchTree` subclass that hooks into `is_finished()`:

```python
class HarnessSearchTree(SearchTree):
    """SearchTree with state-harness Lyapunov monitoring."""
    
    def is_finished(self) -> str | None:
        base_result = super().is_finished()
        if base_result:
            return base_result
        
        # Record token usage for this iteration
        usage = self.total_usage()
        tokens_this_iter = current_total - self._prev_total_tokens
        
        # Check for Lyapunov stability violation
        try:
            self._guard.record_step(tokens_used=tokens_this_iter)
        except StabilityViolation:
            return "harness_stability_violation"  # Early termination
        except BudgetExhausted:
            return "harness_budget_exhausted"
        
        return None
```

This is a **zero-modification** integration — we don't touch the agent's
decision logic, prompt, or tools. We only observe token consumption and
terminate early when the growth-ratio monitor detects a spiral.

### Reproducing Our Results

1. **Clone this repo** and install state-harness
2. **Run `setup.sh`** in each benchmark directory
3. **Run the benchmark** scripts
4. **Compare** baseline vs harness results

The key metrics to verify:
- **Pass rate** (harness ≥ baseline - 2%)
- **Token savings** (MINT: ~1%, τ³-bench: ~9%, SWE-bench: ~50%)
- **Spiral detection** (harness terminates early on spiraling SWE-bench tasks)

---

## Citing

If you use these benchmarks, please cite:

```bibtex
@article{verma2026empirical,
  title={Empirical Lyapunov Stability: Growth-Ratio Energy Functions as Leading Indicators of Agent Task Failure},
  author={Verma, Vishal},
  year={2026},
  url={https://github.com/vishal-dehurdle/state-harness}
}
```
