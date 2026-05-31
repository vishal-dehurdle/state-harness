#!/bin/bash
# ════════════════════════════════════════════════════════════════
# DEFINITIVE BENCHMARK — Sequential domains, concurrency=2
#
# Lesson learned: parallel domains hit Vertex AI 429 rate limits.
# Safe strategy: sequential domains, 2 concurrent tasks per phase.
# 
# 11 phases × 150 runs = 1,650 total runs
# Estimated time: ~20-25 hours (2× faster than concurrency=1)
# ════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TAU3_DIR="$(dirname "$PROJECT_ROOT")/tau3-bench"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results/${TIMESTAMP}_definitive"
mkdir -p "$RESULTS_DIR"

LOG_FILE="${RESULTS_DIR}/benchmark.log"

AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
NUM_TRIALS=3
CONC=2  # Safe concurrency for Vertex AI rate limits

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  DEFINITIVE BENCHMARK v3 — Sequential, concurrency=$CONC" | tee -a "$LOG_FILE"
echo "  Results:  $RESULTS_DIR"                                  | tee -a "$LOG_FILE"
echo "  Phases:   11 (telecom→airline→retail+sweep)"             | tee -a "$LOG_FILE"
echo "  Runs:     1,650 (11 × 50 tasks × 3 trials)"             | tee -a "$LOG_FILE"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"              | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$TAU3_DIR"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

run_phase() {
    local PHASE="$1"
    local DOMAIN="$2"
    local AGENT="$3"
    local OUTPUT="$4"
    local ENV_PREFIX="${5:-}"

    echo "" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] Phase ${PHASE}/11: ${DOMAIN} ${AGENT}" | tee -a "$LOG_FILE"

    START_TIME=$(date +%s)

    env $ENV_PREFIX \
    uv run tau2 run \
        --domain "$DOMAIN" \
        --agent "$AGENT" \
        --agent-llm "$AGENT_LLM" \
        --user-llm "$USER_LLM" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$CONC" \
        --save-to "${RESULTS_DIR}/${OUTPUT}" \
        2>&1 | tee -a "$LOG_FILE"

    END_TIME=$(date +%s)
    MINS=$(( (END_TIME - START_TIME) / 60 ))

    # Infra error check
    python3 -c "
import json, os
rfile = os.path.join('${RESULTS_DIR}/${OUTPUT}', 'results.json')
if os.path.exists(rfile):
    with open(rfile) as f:
        sims = json.load(f)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    print(f'  → {len(sims)} runs, {infra} infra errors ({\"✅\" if infra == 0 else \"❌\"}), {$MINS}min')
" 2>/dev/null | tee -a "$LOG_FILE"
}

# ── TELECOM (validates infra fix) ─────────────────────────────
run_phase 1   telecom  llm_agent        "telecom_llm_agent.json"
run_phase 2   telecom  naive_cap_agent  "telecom_naive_cap_agent.json"
run_phase 3   telecom  harness_agent    "telecom_harness_agent.json"

# ── AIRLINE ───────────────────────────────────────────────────
run_phase 4   airline  llm_agent        "airline_llm_agent.json"
run_phase 5   airline  naive_cap_agent  "airline_naive_cap_agent.json"
run_phase 6   airline  harness_agent    "airline_harness_agent.json"

# ── RETAIL + THRESHOLD SWEEP ─────────────────────────────────
run_phase 7   retail  llm_agent        "retail_llm_agent.json"
run_phase 8   retail  naive_cap_agent  "retail_naive_cap_agent.json"

run_phase 9   retail  harness_agent    "retail_harness_tau2.0.json" \
    "HARNESS_RATIO_THRESHOLD=2.0 HARNESS_BUDGET_GATE=8000"

run_phase 10  retail  harness_agent    "retail_harness_tau2.5.json"

run_phase 11  retail  harness_agent    "retail_harness_tau3.0.json" \
    "HARNESS_RATIO_THRESHOLD=3.0 HARNESS_BUDGET_GATE=12000"

# ── FINAL SUMMARY ────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  DEFINITIVE BENCHMARK COMPLETE"                            | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"

python3 -c "
import json, os
total = infra_total = 0
for name in sorted(os.listdir('$RESULTS_DIR')):
    rfile = os.path.join('$RESULTS_DIR', name, 'results.json')
    if not os.path.exists(rfile): continue
    with open(rfile) as f:
        sims = json.load(f)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    rewards = [(s.get('reward_info') or {}).get('reward', 0) for s in sims]
    pr = sum(1 for r in rewards if r > 0)/len(rewards)*100
    costs = [s.get('agent_cost', 0) or 0 for s in sims]
    avg_cost = sum(costs)/len(costs)
    total += len(sims)
    infra_total += infra
    status = '✅' if infra == 0 else '❌'
    print(f'  {status} {name:<40s} {len(sims):>4} runs  {infra:>3} infra  pass={pr:.1f}%  cost=\${avg_cost:.4f}')
print(f'  {\"─\"*70}')
print(f'  TOTAL: {total} runs, {infra_total} infra errors')
" 2>/dev/null | tee -a "$LOG_FILE"

echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
