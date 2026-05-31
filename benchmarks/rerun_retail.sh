#!/bin/bash
# ================================================================
# Retail Re-run + Threshold Sweep (v2 — with infra fix + per-domain tuning)
# 
# Cost-optimized strategy:
#   Phase 1: Baseline (3 trials = 150 runs)
#   Phase 2: Naive Cap (3 trials = 150 runs)
#   Phase 3: Harness τ=2.0 (1 trial = 50 runs)  — sweep point
#   Phase 4: Harness τ=2.5 (3 trials = 150 runs) — main result
#   Phase 5: Harness τ=3.0 (1 trial = 50 runs)   — sweep point
#
# Total: 550 runs (~₹10,000 estimated)
#
# Prerequisites:
#   - Fixes applied: to_litellm_messages (parallel tool call bug)
#   - Fixes applied: per-domain thresholds in harness_agent.py
#   - .env configured with GOOGLE_CLOUD_PROJECT=your-gcp-project
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TAU3_DIR="$(dirname "$PROJECT_ROOT")/tau3-bench"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results/${TIMESTAMP}_retail_v2"
mkdir -p "$RESULTS_DIR"

LOG_FILE="${PROJECT_ROOT}/benchmark_retail_v2.log"

AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
MAX_CONCURRENCY=1
DOMAIN="retail"

echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  RETAIL RE-RUN + THRESHOLD SWEEP"  | tee -a "$LOG_FILE"
echo "  Results: $RESULTS_DIR"            | tee -a "$LOG_FILE"
echo "  Fixes applied:"                   | tee -a "$LOG_FILE"
echo "    ✓ to_litellm_messages: no content=null on tool-call turns" | tee -a "$LOG_FILE"
echo "    ✓ per-domain thresholds: τ=2.5 for retail (default)" | tee -a "$LOG_FILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"             | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$TAU3_DIR"

# Load env vars
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ── Phase 1: Baseline (no monitor, 3 trials) ──────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/5] Baseline (llm_agent) — 3 trials" | tee -a "$LOG_FILE"
START=$(date +%s)

uv run tau2 run \
    --domain "$DOMAIN" \
    --agent llm_agent \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 3 \
    --max-concurrency $MAX_CONCURRENCY \
    --output "${RESULTS_DIR}/retail_llm_agent.json" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Phase 1 done in $(($(date +%s) - START))s" | tee -a "$LOG_FILE"

# ── Phase 2: Naive Cap (3 trials) ──────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/5] Naive Cap (naive_cap_agent) — 3 trials" | tee -a "$LOG_FILE"
START=$(date +%s)

uv run tau2 run \
    --domain "$DOMAIN" \
    --agent naive_cap_agent \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 3 \
    --max-concurrency $MAX_CONCURRENCY \
    --output "${RESULTS_DIR}/retail_naive_cap_agent.json" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Phase 2 done in $(($(date +%s) - START))s" | tee -a "$LOG_FILE"

# ── Phase 3: Harness τ=2.0 (1 trial, sweep point) ──────────────
# Override the domain default (2.5) with env var
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [3/5] Harness τ=2.0 — 1 trial (sweep)" | tee -a "$LOG_FILE"
START=$(date +%s)

HARNESS_RATIO_THRESHOLD=2.0 HARNESS_BUDGET_GATE=8000 \
uv run tau2 run \
    --domain "$DOMAIN" \
    --agent harness_agent \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 1 \
    --max-concurrency $MAX_CONCURRENCY \
    --output "${RESULTS_DIR}/retail_harness_tau2.0.json" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Phase 3 done in $(($(date +%s) - START))s" | tee -a "$LOG_FILE"

# ── Phase 4: Harness τ=2.5 (3 trials, main result) ─────────────
# Uses the domain default (no env override needed)
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [4/5] Harness τ=2.5 — 3 trials (main)" | tee -a "$LOG_FILE"
START=$(date +%s)

uv run tau2 run \
    --domain "$DOMAIN" \
    --agent harness_agent \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 3 \
    --max-concurrency $MAX_CONCURRENCY \
    --output "${RESULTS_DIR}/retail_harness_tau2.5.json" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Phase 4 done in $(($(date +%s) - START))s" | tee -a "$LOG_FILE"

# ── Phase 5: Harness τ=3.0 (1 trial, sweep point) ──────────────
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [5/5] Harness τ=3.0 — 1 trial (sweep)" | tee -a "$LOG_FILE"
START=$(date +%s)

HARNESS_RATIO_THRESHOLD=3.0 HARNESS_BUDGET_GATE=12000 \
uv run tau2 run \
    --domain "$DOMAIN" \
    --agent harness_agent \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 1 \
    --max-concurrency $MAX_CONCURRENCY \
    --output "${RESULTS_DIR}/retail_harness_tau3.0.json" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Phase 5 done in $(($(date +%s) - START))s" | tee -a "$LOG_FILE"

# ── Summary ────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  RETAIL RE-RUN + SWEEP COMPLETE" | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Results:" | tee -a "$LOG_FILE"
for f in "$RESULTS_DIR"/*.json; do
    echo "    $(basename "$f")" | tee -a "$LOG_FILE"
done
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
