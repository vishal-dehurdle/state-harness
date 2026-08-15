#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
# MINT Full-Stack Benchmark — 4 conditions
#
# Conditions:
#   A. Baseline        — no harness monitoring (HARNESS_MODE=off)
#   B. Lyapunov-Only   — HARNESS_RG=off, HARNESS_VSA=off
#   C. Lyapunov + RG   — HARNESS_VSA=off
#   D. Full-Stack      — all mechanisms on
#
# Tasks: GSM8K (48), MATH (100), HumanEval (45), MBPP (91) = 284 total
# Each condition runs all 284 tasks into its own output directory.
# ════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MINT_DIR="$(cd "$STATE_HARNESS_DIR/../mint-bench" && pwd)"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="$STATE_HARNESS_DIR/benchmark_results/${TIMESTAMP}_mint_fullstack"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

# Source environment
if [ -f "$STATE_HARNESS_DIR/.env" ]; then
    set -a; source "$STATE_HARNESS_DIR/.env"; set +a
fi

cd "$MINT_DIR"

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  MINT FULL-STACK BENCHMARK — 4 Conditions"                 | tee -a "$LOG_FILE"
echo "  Results:  $RESULTS_DIR"                                    | tee -a "$LOG_FILE"
echo "  Tasks:    GSM8K(48) + MATH(100) + HumanEval(45) + MBPP(91) = 284" | tee -a "$LOG_FILE"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

CONFIGS=(
    "gemini_baseline_reasoning_gsm8k.json"
    "gemini_baseline_reasoning_math.json"
    "gemini_baseline_coding_humaneval.json"
    "gemini_baseline_coding_mbpp.json"
)

run_condition() {
    local CONDITION="$1"
    local SUFFIX="$2"
    local ENV_PREFIX="${3:-}"

    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] Condition: ${CONDITION}" | tee -a "$LOG_FILE"
    echo "  Suffix: ${SUFFIX}, Env: ${ENV_PREFIX:-default}" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

    START_TIME=$(date +%s)

    for config in "${CONFIGS[@]}"; do
        CONFIG_PATH="$SCRIPT_DIR/configs/$config"

        if [ ! -f "$CONFIG_PATH" ]; then
            echo "  ⚠️  Config not found: $CONFIG_PATH" | tee -a "$LOG_FILE"
            continue
        fi

        echo "  [$(date '+%H:%M:%S')] Running: $config" | tee -a "$LOG_FILE"

        env $ENV_PREFIX \
        .venv/bin/python "$SCRIPT_DIR/run_harness_mint.py" \
            --exp_config "$CONFIG_PATH" \
            --condition-suffix "$SUFFIX" \
            --fresh \
            2>&1 | tail -5 | tee -a "$LOG_FILE"
    done

    END_TIME=$(date +%s)
    MINS=$(( (END_TIME - START_TIME) / 60 ))
    echo "  ✅ Condition ${CONDITION} complete (${MINS}min)" | tee -a "$LOG_FILE"
}

# ── Condition A: Baseline (no monitoring) ─────────────────────
run_condition "A-Baseline" "A_baseline" "HARNESS_MODE=off"

# ── Condition B: Lyapunov-Only ────────────────────────────────
run_condition "B-Lyapunov" "B_lyapunov" "HARNESS_RG=off HARNESS_VSA=off"

# ── Condition C: Lyapunov + RG ────────────────────────────────
run_condition "C-Lyapunov+RG" "C_lyapunov_rg" "HARNESS_VSA=off"

# ── Condition D: Full-Stack ───────────────────────────────────
run_condition "D-FullStack" "D_fullstack" ""

# ── Summary ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  MINT FULL-STACK BENCHMARK COMPLETE"                        | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "  Results:  $RESULTS_DIR"                                    | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
