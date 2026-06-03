#!/bin/bash
# ════════════════════════════════════════════════════════════════
# MINT LOCAL MODEL BENCHMARK
#
# Runs MINT benchmark (multi-turn interaction with tools) against
# Ollama local models. Tests: Reasoning (GSM8K, MATH) and
# Coding (HumanEval, MBPP).
#
# 2 conditions per model:
#   A. Baseline (no harness monitoring)
#   B. Harness (state-harness monitoring)
#
# Usage:
#   ./benchmarks/local_models/run_mint_local.sh <model>
#   ./benchmarks/local_models/run_mint_local.sh qwen3:4b
#
# Prerequisites:
#   - Ollama running with the model pulled
#   - MINT venv set up (../mint-bench/.venv)
#   - Vertex AI creds for user simulator (or use local model)
# ════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL="${1:?Usage: $0 <ollama-model-name> (e.g., qwen3:4b)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MINT_DIR="$(cd "$PROJECT_ROOT/../mint-bench" && pwd)"

# Sanitize model name for filenames
MODEL_SAFE="${MODEL//:/_}"
MODEL_SAFE="${MODEL_SAFE//\//_}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results/local_models/mint_${MODEL_SAFE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

# Source environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  MINT LOCAL MODEL BENCHMARK"                                | tee -a "$LOG_FILE"
echo "  Model:    ${MODEL} (via Ollama)"                           | tee -a "$LOG_FILE"
echo "  Results:  ${RESULTS_DIR}"                                  | tee -a "$LOG_FILE"
echo "  Tasks:    GSM8K + MATH + HumanEval + MBPP"               | tee -a "$LOG_FILE"
echo "  Phases:   2 (Baseline → Harness)"                         | tee -a "$LOG_FILE"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"

# Verify Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama not running. Start it: ollama serve" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Ollama is running" | tee -a "$LOG_FILE"

cd "$MINT_DIR"

# Generate MINT config files on the fly for this model
CONFIGS_DIR="${RESULTS_DIR}/configs"
mkdir -p "$CONFIGS_DIR"

TASKS=(
    "reasoning_gsm8k:ReasoningTask:reasoning:data/processed/gsm8k/test_prompts.json"
    "reasoning_math:ReasoningTask:reasoning:data/processed/math/test_prompts.json"
    "coding_humaneval:HumanEvalTask:coding:data/processed/humaneval/test_prompts.json"
    "coding_mbpp:MBPPTask:coding:data/processed/mbpp/test_prompts.json"
)

for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r task_name task_class task_type filepath <<< "$task_spec"

    cat > "${CONFIGS_DIR}/${task_name}_baseline.json" << JSONEOF
{
    "agent": {
        "agent_class": "GeminiLMAgent",
        "config": {
            "model_name": "ollama/${MODEL}",
            "chat_mode": true,
            "max_tokens": 512,
            "temperature": 0.0
        }
    },
    "task": {
        "task_class": "${task_class}",
        "task_type": "${task_type}",
        "tool_imports": [],
        "filepath": "${filepath}"
    },
    "output_dir": "data/outputs/local_${MODEL_SAFE}/F=None/max5_p2+tool+cd/${task_type}/${task_name}",
    "env_config": {
        "max_steps": 5,
        "use_tools": true,
        "max_propose_solution": 2,
        "count_down": true
    },
    "feedback_config": {
        "feedback_agent_config": {
            "chat_mode": true,
            "max_tokens": 1024,
            "temperature": 0.0,
            "stop": ["\\nQ:"],
            "agent_class": "None",
            "model_name": "None"
        },
        "pseudo_human_feedback": "None",
        "feedback_form": "None"
    }
}
JSONEOF
done

echo "✅ Generated ${#TASKS[@]} MINT config files" | tee -a "$LOG_FILE"

# ── Run Baseline (no harness) ─────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══ Phase 1: Baseline (no harness monitoring) ═══" | tee -a "$LOG_FILE"

for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r task_name _ _ _ <<< "$task_spec"
    CONFIG="${CONFIGS_DIR}/${task_name}_baseline.json"
    echo "[$(date +%H:%M:%S)] Running baseline: ${task_name}" | tee -a "$LOG_FILE"

    HARNESS_MODE=off \
    .venv/bin/python -m mint.main --exp_config "$CONFIG" \
        2>&1 | tail -5 | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done
echo "✅ Baseline complete" | tee -a "$LOG_FILE"

# ── Run Harness (with monitoring) ─────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══ Phase 2: Harness (with state-harness monitoring) ═══" | tee -a "$LOG_FILE"

for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r task_name _ _ _ <<< "$task_spec"
    CONFIG="${CONFIGS_DIR}/${task_name}_baseline.json"
    echo "[$(date +%H:%M:%S)] Running harness: ${task_name}" | tee -a "$LOG_FILE"

    .venv/bin/python "$PROJECT_ROOT/benchmarks/mint/run_harness_mint.py" \
        --exp_config "$CONFIG" \
        --condition-suffix "harness" \
        --fresh \
        2>&1 | tail -5 | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done
echo "✅ Harness complete" | tee -a "$LOG_FILE"

# ── Summary ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  MINT LOCAL MODEL BENCHMARK COMPLETE"                       | tee -a "$LOG_FILE"
echo "  Model:    ${MODEL}"                                        | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "  Results:  ${MINT_DIR}/data/outputs/local_${MODEL_SAFE}/"  | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
