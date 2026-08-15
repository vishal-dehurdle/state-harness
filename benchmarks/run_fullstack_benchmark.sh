#!/bin/bash
# ════════════════════════════════════════════════════════════════
# FULL-STACK BENCHMARK — 4 conditions × τ³-bench airline
#
# Conditions:
#   A. Baseline        — llm_agent (no harness)
#   B. Lyapunov-Only   — harness_agent, RG=off, VSA=off
#   C. Lyapunov + RG   — harness_agent, VSA=off
#   D. Full-Stack      — harness_agent (default: RG=on, VSA=on)
#
# 4 phases × 50 tasks × 3 trials = 600 total runs
# Estimated time: ~2 hours
# ════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TAU3_DIR="$(dirname "$PROJECT_ROOT")/tau3-bench"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results/${TIMESTAMP}_fullstack"
mkdir -p "$RESULTS_DIR"

LOG_FILE="${RESULTS_DIR}/benchmark.log"

AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
NUM_TRIALS=3
CONC=2

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  FULL-STACK BENCHMARK — 4 Conditions × Airline"            | tee -a "$LOG_FILE"
echo "  Results:  $RESULTS_DIR"                                    | tee -a "$LOG_FILE"
echo "  Phases:   4 (Baseline → Lyapunov → Lyapunov+RG → Full)"  | tee -a "$LOG_FILE"
echo "  Runs:     600 (4 × 50 tasks × 3 trials)"                 | tee -a "$LOG_FILE"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$TAU3_DIR"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

# Ensure Vertex AI env vars are set
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-dehurdle-caic}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-central1}"

run_phase() {
    local PHASE="$1"
    local CONDITION="$2"
    local AGENT="$3"
    local OUTPUT="$4"
    local ENV_PREFIX="${5:-}"

    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] Phase ${PHASE}/4: Condition ${CONDITION}" | tee -a "$LOG_FILE"
    echo "  Agent: ${AGENT}, Env: ${ENV_PREFIX:-none}" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

    START_TIME=$(date +%s)

    env $ENV_PREFIX \
    uv run tau2 run \
        --domain airline \
        --agent "$AGENT" \
        --agent-llm "$AGENT_LLM" \
        --user-llm "$USER_LLM" \
        --num-trials "$NUM_TRIALS" \
        --max-concurrency "$CONC" \
        --save-to "${RESULTS_DIR}/${OUTPUT}" \
        2>&1 | tee -a "$LOG_FILE"

    END_TIME=$(date +%s)
    MINS=$(( (END_TIME - START_TIME) / 60 ))

    # Quick result check
    python3 -c "
import json, os
rfile = os.path.join('${RESULTS_DIR}/${OUTPUT}', 'results.json')
if os.path.exists(rfile):
    with open(rfile) as f:
        sims = json.load(f)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    rewards = [(s.get('reward_info') or {}).get('reward', 0) for s in sims]
    pr = sum(1 for r in rewards if r > 0)/max(len(rewards),1)*100
    costs = [s.get('agent_cost', 0) or 0 for s in sims]
    avg_cost = sum(costs)/max(len(costs),1)
    print(f'  → {len(sims)} runs, {infra} infra errors, pass={pr:.1f}%, cost=\${avg_cost:.4f}, {$MINS}min')
" 2>/dev/null | tee -a "$LOG_FILE"
}

# ── Condition A: Baseline (no harness) ────────────────────────
run_phase 1 "A-Baseline" llm_agent "airline_A_baseline.json"

# ── Condition B: Lyapunov-Only (RG=off, VSA=off) ─────────────
run_phase 2 "B-Lyapunov" harness_agent "airline_B_lyapunov.json" \
    "HARNESS_RG=off HARNESS_VSA=off"

# ── Condition C: Lyapunov + RG (VSA=off) ─────────────────────
run_phase 3 "C-Lyapunov+RG" harness_agent "airline_C_lyapunov_rg.json" \
    "HARNESS_VSA=off"

# ── Condition D: Full-Stack (RG=on, VSA=on) ──────────────────
run_phase 4 "D-FullStack" harness_agent "airline_D_fullstack.json"


# ── FINAL SUMMARY ────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  FULL-STACK BENCHMARK COMPLETE"                            | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"

python3 -c "
import json, os
results_dir = '$RESULTS_DIR'
print()
print(f'  {'Condition':<45s} {'Runs':>5s}  {'Infra':>5s}  {'Pass%':>6s}  {'AvgCost':>9s}')
print(f'  {\"─\"*75}')
for name in sorted(os.listdir(results_dir)):
    rfile = os.path.join(results_dir, name, 'results.json')
    if not os.path.exists(rfile): continue
    with open(rfile) as f:
        sims = json.load(f)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    rewards = [(s.get('reward_info') or {}).get('reward', 0) for s in sims]
    pr = sum(1 for r in rewards if r > 0)/max(len(rewards),1)*100
    costs = [s.get('agent_cost', 0) or 0 for s in sims]
    avg_cost = sum(costs)/max(len(costs),1)
    status = '✅' if infra == 0 else '❌'
    print(f'  {status} {name:<43s} {len(sims):>5}  {infra:>5}  {pr:>5.1f}%  \${avg_cost:>8.4f}')
" 2>/dev/null | tee -a "$LOG_FILE"

echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
