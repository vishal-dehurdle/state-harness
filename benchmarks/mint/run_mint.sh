#!/usr/bin/env bash
# =============================================================================
# MINT Benchmark Setup & Runner for state-harness evaluation
# =============================================================================
#
# Runs MINT benchmark (multi-turn interaction with tools) using Gemini.
# Tests: Reasoning (GSM8K, MATH) and Coding (HumanEval, MBPP)
#
# Prerequisites:
#   - Python 3.12+
#   - VERTEXAI_PROJECT, VERTEXAI_LOCATION, GOOGLE_APPLICATION_CREDENTIALS set
#
# Usage:
#   ./benchmarks/mint/run_mint.sh [--mode baseline|harness|both]
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MINT_DIR="$(cd "$STATE_HARNESS_DIR/../mint-bench" && pwd)"
RESULTS_DIR="$STATE_HARNESS_DIR/benchmark_results/mint"

MODE="${1:-both}"
if [ "$MODE" = "--mode" ]; then
    MODE="${2:-both}"
fi

# Source environment
if [ -f "$STATE_HARNESS_DIR/.env" ]; then
    set -a; source "$STATE_HARNESS_DIR/.env"; set +a
fi

cd "$MINT_DIR"
source .venv/bin/activate

mkdir -p "$RESULTS_DIR"

# MINT task categories we evaluate
TASKS=(
    "reasoning/gsm8k"
    "reasoning/math"
    "coding/humaneval"
    "coding/mbpp"
)

echo "================================================"
echo " MINT Benchmark"
echo " Mode: $MODE"
echo " Tasks: ${#TASKS[@]}"
echo "================================================"
echo ""

# Run baseline (no feedback, standard env, no harness)
if [ "$MODE" = "baseline" ] || [ "$MODE" = "both" ]; then
    echo "═══ Phase 1: Baseline (no harness monitoring) ═══"
    for task in "${TASKS[@]}"; do
        CONFIG="$SCRIPT_DIR/configs/gemini_baseline_${task//\//_}.json"
        echo "[$(date +%H:%M:%S)] Running baseline: $task"
        python3 -m mint.main --exp_config "$CONFIG" 2>&1 | tail -3
        echo "---"
    done
    echo "✅ Baseline complete"
fi

# Run harness (with state-harness monitoring)
if [ "$MODE" = "harness" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "═══ Phase 2: Harness (with state-harness monitoring) ═══"
    for task in "${TASKS[@]}"; do
        CONFIG="$SCRIPT_DIR/configs/gemini_harness_${task//\//_}.json"
        echo "[$(date +%H:%M:%S)] Running harness: $task"
        python3 -m mint.main --exp_config "$CONFIG" 2>&1 | tail -3
        echo "---"
    done
    echo "✅ Harness complete"
fi

echo ""
echo "================================================"
echo " MINT Benchmark complete!"
echo "================================================"
echo " Results: $MINT_DIR/data/outputs/gemini-*/"
echo " Analysis: python3 $SCRIPT_DIR/analyze_mint_results.py"
echo ""
