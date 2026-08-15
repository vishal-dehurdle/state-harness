#!/usr/bin/env bash
# =============================================================================
# SWE-bench Phases B, C, E — Supplemental to the A+D run
# =============================================================================
#
# Runs conditions:
#   B = Lyapunov-only (harness monitoring, no adaptive RG)
#   C = Lyapunov+RG   (harness monitoring, with adaptive RG — same as D since
#                       VSA is not implemented in SWE-bench integration)
#   E = Naive Cap      (hard 100K token budget, no monitoring)
#
# NOTE: C ≡ D for SWE-bench (no VSA in the moatless integration).
#       We include C for completeness but its results should match Phase D.
#
# Prerequisites:
#   - Docker/OrbStack running
#   - MOATLESS_DIR, VERTEXAI vars set
#   - Phases A + D already completed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOATLESS_DIR_PATH="${MOATLESS_DIR:-$(cd "$STATE_HARNESS_DIR/../moatless-tools" && pwd)/.moatless}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$STATE_HARNESS_DIR/benchmark_results/swe_bench_bce_${TIMESTAMP}.log"
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
echo "  SWE-bench PHASES B, C, E"                                  | tee -a "$LOG_FILE"
echo "  Instances: ${#INSTANCES[@]}"                                | tee -a "$LOG_FILE"
echo "  Concurrency: 1 (sequential)"                               | tee -a "$LOG_FILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"                 | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

run_instance() {
    local EVAL_PREFIX="$1"
    local FLOW="$2"
    local INSTANCE="$3"
    local EVAL_NAME="${EVAL_PREFIX}_${INSTANCE}"

    echo "  [$(date '+%H:%M:%S')] $EVAL_NAME" | tee -a "$LOG_FILE"

    # Remove old evaluation data if exists
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
    local PHASE_NUM="$1"
    local PHASE_LABEL="$2"
    local EVAL_PREFIX="$3"
    local FLOW="$4"

    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] Phase $PHASE_NUM: $PHASE_LABEL" | tee -a "$LOG_FILE"
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

# ── Phase B: Lyapunov-only ───────────────────────────────────
run_phase "B" "Lyapunov-only (no RG decimation)" "swe_ly" "swebench_harness_lyapunov"

# ── Phase E: Naive Cap ───────────────────────────────────────
# Run E before C since C ≡ D (more useful data first)
run_phase "E" "Naive Cap (hard 100K token budget)" "swe_nc" "swebench_naive_cap"

# ── Summary ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  SWE-bench PHASES B, E COMPLETE"                            | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "  Note: Phase C omitted (identical to Phase D for SWE-bench" | tee -a "$LOG_FILE"
echo "        since VSA is not in the moatless integration)"       | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
