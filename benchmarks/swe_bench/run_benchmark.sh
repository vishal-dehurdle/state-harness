#!/usr/bin/env bash
# =============================================================================
# SWE-bench Benchmark Runner
# =============================================================================
#
# Runs the SWE-bench verified_mini benchmark (50 tasks) in two modes:
#   1. Baseline — standard moatless SearchTree (no monitoring)
#   2. Harness  — HarnessSearchTree with state-harness Lyapunov monitoring
#
# Prerequisites:
#   - Run setup.sh first
#   - Docker/OrbStack running
#   - Environment variables set:
#     VOYAGE_API_KEY, VERTEXAI_PROJECT, VERTEXAI_LOCATION,
#     GOOGLE_APPLICATION_CREDENTIALS
#
# Usage:
#   ./benchmarks/swe_bench/run_benchmark.sh [--mode baseline|harness|both] [--concurrency N]
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOATLESS_DIR="$(cd "$STATE_HARNESS_DIR/../moatless-tools" && pwd)"
RESULTS_DIR="$STATE_HARNESS_DIR/benchmark_results/swe_bench"

# Parse arguments
MODE="${1:---mode}"
if [ "$MODE" = "--mode" ]; then
    MODE="${2:-both}"
fi
CONCURRENCY="${3:-3}"

# Source environment
if [ -f "$STATE_HARNESS_DIR/.env" ]; then
    set -a; source "$STATE_HARNESS_DIR/.env"; set +a
fi

cd "$MOATLESS_DIR"
source .venv/bin/activate

mkdir -p "$RESULTS_DIR"

# SWE-bench verified_mini instances (50 tasks from the verified dataset)
# These are the standard benchmark instances used in moatless evaluations
INSTANCES=(
    "django__django-11099"
    "django__django-11283"
    "django__django-11422"
    "django__django-11620"
    "django__django-11797"
    "django__django-11848"
    "django__django-11905"
    "django__django-11910"
    "django__django-11964"
    "django__django-12113"
    "django__django-12125"
    "django__django-12184"
    "django__django-12308"
    "django__django-12470"
    "django__django-12589"
    "django__django-12700"
    "django__django-12708"
    "django__django-12915"
    "django__django-13033"
    "django__django-13158"
    "django__django-13220"
    "django__django-13230"
    "django__django-13315"
    "django__django-13401"
    "django__django-13447"
    "django__django-13551"
    "django__django-13590"
    "django__django-13710"
    "django__django-13757"
    "django__django-13768"
    "django__django-13964"
    "django__django-14016"
    "django__django-14017"
    "django__django-14155"
    "django__django-14238"
    "django__django-14382"
    "django__django-14411"
    "django__django-14534"
    "django__django-14580"
    "django__django-14672"
    "django__django-14730"
    "django__django-14752"
    "django__django-14787"
    "django__django-14855"
    "django__django-14915"
    "django__django-14997"
    "django__django-15061"
    "django__django-15104"
    "django__django-15202"
    "django__django-15213"
)

run_evaluation() {
    local eval_prefix="$1"
    local flow="$2"
    local instance="$3"
    
    # Each instance needs a unique evaluation name
    local eval_name="${eval_prefix}_${instance}"
    
    echo "[$(date +%H:%M:%S)] Starting $eval_name"
    
    MOATLESS_DIR="$(pwd)/.moatless" \
    python3 scripts/docker_run.py \
        --flow "$flow" \
        --litellm-model-name vertex_ai/gemini-2.5-flash \
        --instance-id "$instance" \
        --evaluation-name "$eval_name" 2>&1 | tail -5
    
    echo "[$(date +%H:%M:%S)] Finished $eval_name"
}

echo "================================================"
echo " SWE-bench Benchmark"
echo " Mode: $MODE"
echo " Tasks: ${#INSTANCES[@]}"
echo " Concurrency: $CONCURRENCY"
echo "================================================"
echo ""

# Run baseline
if [ "$MODE" = "baseline" ] || [ "$MODE" = "both" ]; then
    echo "═══ Phase 1: Baseline (no monitoring) ═══"
    EVAL_NAME="swe_baseline_$(date +%Y%m%d)"
    
    RUNNING=0
    for instance in "${INSTANCES[@]}"; do
        run_evaluation "$EVAL_NAME" "swebench_react" "$instance" &
        RUNNING=$((RUNNING + 1))
        
        if [ $RUNNING -ge $CONCURRENCY ]; then
            wait -n
            RUNNING=$((RUNNING - 1))
        fi
    done
    wait
    echo "✅ Baseline complete"
fi

# Run harness
if [ "$MODE" = "harness" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "═══ Phase 2: Harness (with state-harness monitoring) ═══"
    EVAL_NAME="swe_harness_$(date +%Y%m%d)"
    
    RUNNING=0
    for instance in "${INSTANCES[@]}"; do
        run_evaluation "$EVAL_NAME" "swebench_harness" "$instance" &
        RUNNING=$((RUNNING + 1))
        
        if [ $RUNNING -ge $CONCURRENCY ]; then
            wait -n
            RUNNING=$((RUNNING - 1))
        fi
    done
    wait
    echo "✅ Harness complete"
fi

echo ""
echo "================================================"
echo " Benchmark complete!"
echo "================================================"
echo " Results saved to: $RESULTS_DIR"
echo " Trajectory data: $MOATLESS_DIR/.moatless/trajs/"
echo ""
echo " To analyze results:"
echo "   python3 $SCRIPT_DIR/analyze_swe_results.py"
echo ""
