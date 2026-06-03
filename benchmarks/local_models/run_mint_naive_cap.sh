#!/bin/bash
# ════════════════════════════════════════════════════════════════
# MINT NAIVE CAP PHASE - Runs after baseline + harness complete
# Naive cap = max_steps=2 (hard cutoff at 2 turns)
# ════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL="${1:?Usage: $0 <ollama-model-name>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MINT_DIR="$(cd "$PROJECT_ROOT/../mint-bench" && pwd)"

MODEL_SAFE="${MODEL//:/_}"
MODEL_SAFE="${MODEL_SAFE//\//_}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results/local_models/mint_naive_${MODEL_SAFE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  MINT NAIVE CAP BENCHMARK"                                  | tee -a "$LOG_FILE"
echo "  Model:    ${MODEL} (via Ollama)"                           | tee -a "$LOG_FILE"
echo "  Config:   max_steps=2 (naive cap)"                         | tee -a "$LOG_FILE"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"

cd "$MINT_DIR"

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

    cat > "${CONFIGS_DIR}/${task_name}_naive.json" << JSONEOF
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
    "output_dir": "data/outputs/local_${MODEL_SAFE}/F=None/max2_naive/${task_type}/${task_name}",
    "env_config": {
        "max_steps": 2,
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

echo "✅ Generated ${#TASKS[@]} naive cap configs (max_steps=2)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "═══ Naive Cap (max_steps=2) ═══" | tee -a "$LOG_FILE"

for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r task_name _ _ _ <<< "$task_spec"
    CONFIG="${CONFIGS_DIR}/${task_name}_naive.json"
    echo "[$(date +%H:%M:%S)] Running naive cap: ${task_name}" | tee -a "$LOG_FILE"

    HARNESS_MODE=off \
    .venv/bin/python -m mint.main --exp_config "$CONFIG" \
        2>&1 | tail -5 | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  MINT NAIVE CAP COMPLETE"                                   | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "  Results:  ${MINT_DIR}/data/outputs/local_${MODEL_SAFE}/"  | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
