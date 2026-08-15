#!/usr/bin/env bash
# =============================================================================
# SWE-bench Multi-Trial Benchmark — 3 trials × 3 conditions (A, D, E)
# =============================================================================
#
# Runs 3 independent trials per instance for conditions:
#   A = Baseline (no monitoring)
#   D = Full-stack harness (Lyapunov + RG + VSA)
#   E = Naive Cap (hard budget, no harness)
#
# This eliminates nondeterminism and gives us error bars.
#
# Infra safeguards:
#   - Docker container + image prune after EVERY instance
#   - Old moatless project data cleaned before each run
#   - Concurrency = 1 (sequential)
#   - VOYAGE_API_KEY loaded from .env.local
#   - Disk check every 10 instances
#
# Total runs: 37 instances × 3 trials × 3 conditions = 333 runs
# Estimated: ~12-15 hours
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOATLESS_DIR_PATH="${MOATLESS_DIR:-$(cd "$STATE_HARNESS_DIR/../moatless-tools" && pwd)/.moatless}"
MOATLESS_ROOT="$(cd "$STATE_HARNESS_DIR/../moatless-tools" && pwd)"
MOATLESS_VENV="$MOATLESS_ROOT/.venv/bin"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="$STATE_HARNESS_DIR/benchmark_results/swe_multi_trial_${TIMESTAMP}"
LOG_FILE="$RESULTS_DIR/run.log"
mkdir -p "$RESULTS_DIR"

# ── Load environment ─────────────────────────────────────────
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-dehurdle-caic}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-asia-south1}"

# Load Voyage key from moatless .env.local
if [ -f "$MOATLESS_ROOT/.env.local" ]; then
    source <(grep VOYAGE_API_KEY "$MOATLESS_ROOT/.env.local" | sed 's/^/export /')
    echo "✅ VOYAGE_API_KEY loaded from .env.local" | tee "$LOG_FILE"
else
    echo "⚠️  No .env.local found — Voyage may fail" | tee "$LOG_FILE"
fi

# Verify env
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT" | tee -a "$LOG_FILE"
echo "VERTEXAI_LOCATION=$VERTEXAI_LOCATION" | tee -a "$LOG_FILE"
echo "VOYAGE_API_KEY=${VOYAGE_API_KEY:+SET}" | tee -a "$LOG_FILE"

# ── Only the 37 instances that have Docker eval images ───────
INSTANCES=(
    "django__django-11099" "django__django-11283" "django__django-11422"
    "django__django-11620" "django__django-11797" "django__django-11848"
    "django__django-11905" "django__django-11910" "django__django-11964"
    "django__django-12113" "django__django-12125" "django__django-12184"
    "django__django-12308" "django__django-12470" "django__django-12589"
    "django__django-12700" "django__django-12708" "django__django-12915"
    "django__django-13033" "django__django-13158"
    "django__django-13315" "django__django-13401"
    "django__django-13551" "django__django-13590"
    "django__django-13757"
    "django__django-13964" "django__django-14017"
    "django__django-14155" "django__django-14238"
    "django__django-14534" "django__django-14580"
    "django__django-14672" "django__django-14752"
    "django__django-14787" "django__django-14855" "django__django-14915"
    "django__django-15104"
)

NUM_TRIALS=3

# Conditions: A=baseline, D=full-stack, E=naive cap
get_flow() {
    case "$1" in
        A) echo "swebench_react" ;;
        D) echo "swebench_harness" ;;
        E) echo "swebench_naive_cap" ;;
    esac
}

CONDITION_ORDER="A D E"

NUM_CONDITIONS=3
TOTAL_RUNS=$(( ${#INSTANCES[@]} * NUM_TRIALS * NUM_CONDITIONS ))

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  SWE-bench MULTI-TRIAL BENCHMARK"                          | tee -a "$LOG_FILE"
echo "  Instances: ${#INSTANCES[@]}"                               | tee -a "$LOG_FILE"
echo "  Trials: $NUM_TRIALS"                                       | tee -a "$LOG_FILE"
echo "  Conditions: $CONDITION_ORDER"                               | tee -a "$LOG_FILE"
echo "  Total runs: $TOTAL_RUNS"                                   | tee -a "$LOG_FILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"                | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

run_instance() {
    local EVAL_PREFIX="$1"
    local FLOW="$2"
    local INSTANCE="$3"
    local EVAL_NAME="${EVAL_PREFIX}_${INSTANCE}"

    # Remove old evaluation data
    rm -rf "$MOATLESS_DIR_PATH/projects/$EVAL_NAME" 2>/dev/null || true

    cd "$MOATLESS_ROOT"
    MOATLESS_DIR="$MOATLESS_DIR_PATH" \
    "$MOATLESS_VENV/python3" scripts/docker_run.py \
        --flow "$FLOW" \
        --litellm-model-name vertex_ai/gemini-2.5-flash \
        --instance-id "$INSTANCE" \
        --evaluation-name "$EVAL_NAME" \
        --use-local 2>&1 | tail -5
    
    local EXIT_CODE=${PIPESTATUS[0]}
    cd "$STATE_HARNESS_DIR"
    return $EXIT_CODE
}

check_disk() {
    local AVAIL=$(df -g / | tail -1 | awk '{print $4}')
    if [ "$AVAIL" -lt 10 ]; then
        echo "  ⚠️  LOW DISK: ${AVAIL}GB available. Running aggressive cleanup..." | tee -a "$LOG_FILE"
        docker system prune -af --volumes > /dev/null 2>&1 || true
        local NEW_AVAIL=$(df -g / | tail -1 | awk '{print $4}')
        echo "  → After cleanup: ${NEW_AVAIL}GB available" | tee -a "$LOG_FILE"
        if [ "$NEW_AVAIL" -lt 5 ]; then
            echo "  🔴 CRITICAL: Only ${NEW_AVAIL}GB left. Aborting." | tee -a "$LOG_FILE"
            exit 1
        fi
    fi
}

GLOBAL_RUN=0
GLOBAL_ERRORS=0

for CONDITION in $CONDITION_ORDER; do
    FLOW=$(get_flow "$CONDITION")
    
    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] Condition $CONDITION ($FLOW)" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    
    PHASE_START=$(date +%s)
    PHASE_ERRORS=0
    PHASE_RESOLVED=0
    INSTANCE_NUM=0
    
    for TRIAL in $(seq 1 $NUM_TRIALS); do
        echo "" | tee -a "$LOG_FILE"
        echo "  ── Trial $TRIAL/$NUM_TRIALS ──" | tee -a "$LOG_FILE"
        
        for instance in "${INSTANCES[@]}"; do
            GLOBAL_RUN=$((GLOBAL_RUN + 1))
            INSTANCE_NUM=$((INSTANCE_NUM + 1))
            EVAL_PREFIX="mt_${CONDITION}_t${TRIAL}"
            
            echo "  [$GLOBAL_RUN/$TOTAL_RUNS] $EVAL_PREFIX $instance" | tee -a "$LOG_FILE"
            
            if run_instance "$EVAL_PREFIX" "$FLOW" "$instance" >> "$LOG_FILE" 2>&1; then
                # Check if resolved
                RESULT_DIR="$MOATLESS_DIR_PATH/projects/${EVAL_PREFIX}_${instance}"
                if [ -f "$RESULT_DIR/trajectory.json" ]; then
                    RESOLVED=$($MOATLESS_VENV/python3 -c "
import json
try:
    with open('$RESULT_DIR/trajectory.json') as f:
        d = json.load(f)
    print('yes' if d.get('info', {}).get('resolved', False) else 'no')
except: print('unknown')
" 2>/dev/null || echo "unknown")
                    if [ "$RESOLVED" = "yes" ]; then
                        PHASE_RESOLVED=$((PHASE_RESOLVED + 1))
                        echo "    ✅ Resolved" | tee -a "$LOG_FILE"
                    else
                        echo "    ❌ Not resolved" | tee -a "$LOG_FILE"
                    fi
                fi
            else
                PHASE_ERRORS=$((PHASE_ERRORS + 1))
                GLOBAL_ERRORS=$((GLOBAL_ERRORS + 1))
                echo "    ⚠️  Error" | tee -a "$LOG_FILE"
            fi
            
            # Cleanup after every instance
            docker container prune -f > /dev/null 2>&1 || true
            
            # Disk check every 10 instances
            if [ $((INSTANCE_NUM % 10)) -eq 0 ]; then
                check_disk
            fi
        done
    done
    
    PHASE_END=$(date +%s)
    PHASE_MINS=$(( (PHASE_END - PHASE_START) / 60 ))
    TOTAL_PHASE_RUNS=$(( ${#INSTANCES[@]} * NUM_TRIALS ))
    
    echo "" | tee -a "$LOG_FILE"
    echo "  → Condition $CONDITION: $TOTAL_PHASE_RUNS runs, $PHASE_RESOLVED resolved, $PHASE_ERRORS errors, ${PHASE_MINS}min" | tee -a "$LOG_FILE"
done

# ── Final Summary ────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  SWE-bench MULTI-TRIAL COMPLETE"                            | tee -a "$LOG_FILE"
echo "  Total: $GLOBAL_RUN runs, $GLOBAL_ERRORS errors"           | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "To analyze: python3 benchmarks/swe_bench/analyze_multi_trial.py $RESULTS_DIR" | tee -a "$LOG_FILE"
