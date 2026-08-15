#!/bin/bash
# ════════════════════════════════════════════════════════════════
# τ³-bench LOCAL MODEL BENCHMARK
#
# Runs τ³-bench airline tasks against Ollama local models.
# 3 conditions per model:
#   A. Baseline (llm_agent, no harness)
#   B. Naive Cap (naive_cap_agent)
#   C. Harness (harness_agent with state-harness)
#
# Usage:
#   ./benchmarks/local_models/run_tau3_local.sh <model>
#   ./benchmarks/local_models/run_tau3_local.sh qwen3:4b
#   ./benchmarks/local_models/run_tau3_local.sh llama3.2:3b
#   ./benchmarks/local_models/run_tau3_local.sh phi4-mini
#   ./benchmarks/local_models/run_tau3_local.sh gemma4:e4b
#
# Prerequisites:
#   - Ollama running with the model pulled
#   - τ³-bench venv set up (../tau3-bench/.venv)
#   - state-harness installed in τ³-bench venv
# ════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL="${1:?Usage: $0 <ollama-model-name> (e.g., qwen3:4b)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAU3_DIR="$(cd "$PROJECT_ROOT/../tau3-bench" && pwd)"

# Sanitize model name for filenames
MODEL_SAFE="${MODEL//:/_}"
MODEL_SAFE="${MODEL_SAFE//\//_}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results/local_models/tau3_${MODEL_SAFE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

# The agent LLM uses litellm's ollama provider
AGENT_LLM="ollama/${MODEL}"

# User simulator still uses cloud model (it generates realistic user responses)
# If you want fully local, change this too — but user sim quality matters less
USER_LLM="vertex_ai/gemini-2.5-flash"

# Fewer trials for local models (they're slow)
NUM_TRIALS=1
CONC=1  # Sequential — local models can't handle parallel requests

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  τ³-bench LOCAL MODEL BENCHMARK"                           | tee -a "$LOG_FILE"
echo "  Model:    ${MODEL} (via Ollama)"                           | tee -a "$LOG_FILE"
echo "  Agent:    ${AGENT_LLM}"                                    | tee -a "$LOG_FILE"
echo "  User:     ${USER_LLM}"                                    | tee -a "$LOG_FILE"
echo "  Results:  ${RESULTS_DIR}"                                  | tee -a "$LOG_FILE"
echo "  Phases:   3 (Baseline → Naive Cap → Harness)"            | tee -a "$LOG_FILE"
echo "  Trials:   ${NUM_TRIALS}"                                   | tee -a "$LOG_FILE"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"

# Verify Ollama is running and model is available
if ! curl -s http://localhost:11434/api/tags | grep -q "${MODEL}"; then
    echo "❌ Model '${MODEL}' not found in Ollama. Pull it first:" | tee -a "$LOG_FILE"
    echo "   ollama pull ${MODEL}" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Ollama model '${MODEL}' is available" | tee -a "$LOG_FILE"

cd "$TAU3_DIR"

# Source environment (for user sim Vertex AI creds)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-dehurdle-caic}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-central1}"

run_phase() {
    local PHASE="$1"
    local CONDITION="$2"
    local AGENT="$3"
    local OUTPUT="$4"
    local ENV_PREFIX="${5:-}"

    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] Phase ${PHASE}/3: Condition ${CONDITION}" | tee -a "$LOG_FILE"
    echo "  Agent: ${AGENT}, Model: ${AGENT_LLM}" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

    START_TIME=$(date +%s)

    env $ENV_PREFIX \
    uv run tau2 run \
        --domain airline \
        --agent "$AGENT" \
        --agent-llm "$AGENT_LLM" \
        --user-llm "$USER_LLM" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$CONC" \
        --save-to "${RESULTS_DIR}/${OUTPUT}" \
        2>&1 | tee -a "$LOG_FILE"

    END_TIME=$(date +%s)
    MINS=$(( (END_TIME - START_TIME) / 60 ))

    # Quick result check
    python3 -c "
import json, os
rfile = os.path.join('${RESULTS_DIR}/${OUTPUT}', 'results.json')
if os.path.exists(rfile):
    with open(rfile) as f:
        sims = json.load(f)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    rewards = [(s.get('reward_info') or {}).get('reward', 0) for s in sims]
    pr = sum(1 for r in rewards if r > 0)/max(len(rewards),1)*100
    costs = [s.get('agent_cost', 0) or 0 for s in sims]
    avg_cost = sum(costs)/max(len(costs),1)
    print(f'  → {len(sims)} runs, {infra} infra errors, pass={pr:.1f}%, cost=\${avg_cost:.4f}, ${MINS}min')
" 2>/dev/null | tee -a "$LOG_FILE"
}

# ── Condition A: Baseline (no harness) ────────────────────────
run_phase 1 "A-Baseline" llm_agent "airline_A_baseline"

# ── Condition B: Naive Cap ────────────────────────────────────
run_phase 2 "B-NaiveCap" naive_cap_agent "airline_B_naive_cap"

# ── Condition C: Harness (full-stack monitoring) ──────────────
run_phase 3 "C-Harness" harness_agent "airline_C_harness"


# ── FINAL SUMMARY ────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  τ³-bench LOCAL MODEL BENCHMARK COMPLETE"                   | tee -a "$LOG_FILE"
echo "  Model:    ${MODEL}"                                        | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
