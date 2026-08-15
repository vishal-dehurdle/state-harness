#!/usr/bin/env bash
# =============================================================================
# SWE-bench Full-Stack Benchmark — Sequential, concurrency=1
# =============================================================================
#
# Runs baseline and harness modes sequentially, one instance at a time.
# This avoids Docker pressure, rate limits, and disk issues.
#
# Prerequisites:
#   - Docker/OrbStack running
#   - MOATLESS_DIR, VERTEXAI vars set
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOATLESS_DIR_PATH="${MOATLESS_DIR:-$(cd "$STATE_HARNESS_DIR/../moatless-tools" && pwd)/.moatless}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$STATE_HARNESS_DIR/benchmark_results/swe_bench_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

# 50 Django instances from SWE-bench verified_mini
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
echo "  SWE-bench FULL-STACK BENCHMARK"                           | tee -a "$LOG_FILE"
echo "  Instances: ${#INSTANCES[@]}"                               | tee -a "$LOG_FILE"
echo "  Concurrency: 1 (sequential)"                              | tee -a "$LOG_FILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"                | tee -a "$LOG_FILE"
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

# ── Phase 1: Baseline ────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Phase 1/2: Baseline (no monitoring)" | tee -a "$LOG_FILE"
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

PHASE1_START=$(date +%s)
COMPLETED=0
ERRORS=0

for instance in "${INSTANCES[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo "  [$COMPLETED/${#INSTANCES[@]}]" | tee -a "$LOG_FILE"
    if ! run_instance "swe_bl" "swebench_react" "$instance"; then
        ERRORS=$((ERRORS + 1))
        echo "  ⚠️  Error on $instance" | tee -a "$LOG_FILE"
    fi
    # Clean up Docker containers between runs to save disk
    docker container prune -f > /dev/null 2>&1 || true
done

PHASE1_END=$(date +%s)
PHASE1_MIN=$(( (PHASE1_END - PHASE1_START) / 60 ))
echo "  → Baseline: ${#INSTANCES[@]} instances, $ERRORS errors, ${PHASE1_MIN}min" | tee -a "$LOG_FILE"

# ── Phase 2: Harness ─────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Phase 2/2: Harness (with monitoring)" | tee -a "$LOG_FILE"
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

PHASE2_START=$(date +%s)
COMPLETED=0
ERRORS=0

for instance in "${INSTANCES[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo "  [$COMPLETED/${#INSTANCES[@]}]" | tee -a "$LOG_FILE"
    if ! run_instance "swe_hr" "swebench_harness" "$instance"; then
        ERRORS=$((ERRORS + 1))
        echo "  ⚠️  Error on $instance" | tee -a "$LOG_FILE"
    fi
    docker container prune -f > /dev/null 2>&1 || true
done

PHASE2_END=$(date +%s)
PHASE2_MIN=$(( (PHASE2_END - PHASE2_START) / 60 ))
echo "  → Harness: ${#INSTANCES[@]} instances, $ERRORS errors, ${PHASE2_MIN}min" | tee -a "$LOG_FILE"

# ── Summary ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  SWE-bench BENCHMARK COMPLETE"                              | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "  Baseline: ${PHASE1_MIN}min, Harness: ${PHASE2_MIN}min"   | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
