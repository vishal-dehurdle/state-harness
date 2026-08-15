#!/usr/bin/env bash
# =============================================================================
# SWE-bench Phases D (rerun), B, E — with --use-local for harness_loop.py
# =============================================================================
#
# Runs:
#   D = Full-stack harness (Lyapunov + adaptive RG) — RERUN with source mount
#   B = Lyapunov-only (no adaptive RG)
#   E = Naive Cap (hard budget, no harness monitoring)
#
# Baseline (A) already completed successfully with 14/37 resolved.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOATLESS_DIR_PATH="${MOATLESS_DIR:-$(cd "$STATE_HARNESS_DIR/../moatless-tools" && pwd)/.moatless}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$STATE_HARNESS_DIR/benchmark_results/swe_bench_dbe_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Same 50 Django instances
INSTANCES=(
    "django__django-11099" "django__django-11283" "django__django-11422"
    "django__django-11620" "django__django-11797" "django__django-11848"
    "django__django-11905" "django__django-11910" "django__django-11964"
    "django__django-12113" "django__django-12125" "django__django-12184"
    "django__django-12308" "django__django-12470" "django__django-12589"
    "django__django-12700" "django__django-12708" "django__django-12915"
    "django__django-13033" "django__django-13158" "django__django-13220"
    "django__django-13230" "django__django-13315" "django__django-13401"
    "django__django-13447" "django__django-13551" "django__django-13590"
    "django__django-13710" "django__django-13757" "django__django-13768"
    "django__django-13964" "django__django-14016" "django__django-14017"
    "django__django-14155" "django__django-14238" "django__django-14382"
    "django__django-14411" "django__django-14534" "django__django-14580"
    "django__django-14672" "django__django-14730" "django__django-14752"
    "django__django-14787" "django__django-14855" "django__django-14915"
    "django__django-14997" "django__django-15061" "django__django-15104"
    "django__django-15202" "django__django-15213"
)

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  SWE-bench PHASES D (rerun), B, E"                          | tee -a "$LOG_FILE"
echo "  Instances: ${#INSTANCES[@]}"                                | tee -a "$LOG_FILE"
echo "  Source mount: --use-local (harness_loop.py)"               | tee -a "$LOG_FILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"                 | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

run_instance() {
    local EVAL_PREFIX="$1"
    local FLOW="$2"
    local INSTANCE="$3"
    local EVAL_NAME="${EVAL_PREFIX}_${INSTANCE}"

    echo "  [$(date '+%H:%M:%S')] $EVAL_NAME" | tee -a "$LOG_FILE"

    rm -rf "$MOATLESS_DIR_PATH/projects/$EVAL_NAME" 2>/dev/null || true

    MOATLESS_DIR="$MOATLESS_DIR_PATH" \
    python3 scripts/docker_run.py \
        --flow "$FLOW" \
        --litellm-model-name vertex_ai/gemini-2.5-flash \
        --instance-id "$INSTANCE" \
        --evaluation-name "$EVAL_NAME" \
        --use-local 2>&1 | tail -3 | tee -a "$LOG_FILE"

    echo "  [$(date '+%H:%M:%S')] Done: $EVAL_NAME" | tee -a "$LOG_FILE"
}

run_phase() {
    local PHASE_LABEL="$1"
    local EVAL_PREFIX="$2"
    local FLOW="$3"

    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] $PHASE_LABEL" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

    local START=$(date +%s)
    local COMPLETED=0
    local ERRORS=0

    for instance in "${INSTANCES[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        echo "  [$COMPLETED/${#INSTANCES[@]}]" | tee -a "$LOG_FILE"
        if ! run_instance "$EVAL_PREFIX" "$FLOW" "$instance"; then
            ERRORS=$((ERRORS + 1))
            echo "  ⚠️  Error on $instance" | tee -a "$LOG_FILE"
        fi
        docker container prune -f > /dev/null 2>&1 || true
    done

    local END=$(date +%s)
    local MINS=$(( (END - START) / 60 ))
    echo "  → $PHASE_LABEL: ${#INSTANCES[@]} instances, $ERRORS errors, ${MINS}min" | tee -a "$LOG_FILE"
}

# ── Phase D: Full-stack harness (rerun with source mount) ────
run_phase "Phase D: Full-stack (Lyapunov + adaptive RG)" "swe_hr" "swebench_harness"

# ── Phase B: Lyapunov-only ───────────────────────────────────
run_phase "Phase B: Lyapunov-only (no RG decimation)" "swe_ly" "swebench_harness_lyapunov"

# ── Phase E: Naive Cap ───────────────────────────────────────
run_phase "Phase E: Naive Cap (hard budget)" "swe_nc" "swebench_naive_cap"

# ── Summary ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  SWE-bench PHASES D, B, E COMPLETE"                         | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
