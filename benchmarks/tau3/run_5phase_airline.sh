#!/bin/bash
# =============================================================================
# τ³-bench Airline — 5-Phase Clean Rerun (concurrency=1)
# =============================================================================
#
# Phases:
#   A = Baseline (llm_agent, no monitoring)
#   B = Lyapunov-only (harness_agent, HARNESS_RG=off HARNESS_VSA=off)
#   C = Lyapunov+RG (harness_agent, HARNESS_VSA=off)
#   D = Full-stack (harness_agent, default — Lyapunov + RG + VSA)
#   E = Naive Cap (naive_cap_agent)
#
# Concurrency=1 to avoid Gemini API tool_call_id corruption.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAU3_DIR="$(cd "$PROJECT_ROOT/../tau3-bench" && pwd)"

# Source env vars
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
CONC=1  # Sequential to avoid infra errors
TRIALS=3

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_DIR="$PROJECT_ROOT/benchmark_results/tau3_5phase_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/run.log"

cd "$TAU3_DIR"

echo "═══════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  τ³-bench AIRLINE — 5-Phase Clean Rerun"                    | tee -a "$LOG_FILE"
echo "  Concurrency: $CONC (sequential)"                           | tee -a "$LOG_FILE"
echo "  Trials: $TRIALS per task"                                  | tee -a "$LOG_FILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"                 | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

run_phase() {
    local PHASE_LABEL="$1"
    local AGENT="$2"
    local OUTPUT_FILE="$3"
    shift 3
    # remaining args are env var overrides

    echo "" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] $PHASE_LABEL" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

    local START=$(date +%s)

    env "$@" uv run tau2 run --domain airline --agent "$AGENT" \
        --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
        --num-trials $TRIALS --max-concurrency $CONC \
        --save-to "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"

    local END=$(date +%s)
    local MINS=$(( (END - START) / 60 ))
    echo "[$(date '+%H:%M:%S')] → $PHASE_LABEL: ${MINS}min" | tee -a "$LOG_FILE"
}

# ── Phase A: Baseline ────────────────────────────────────────
run_phase "Phase A: Baseline (no monitoring)" \
    "llm_agent" "$RESULTS_DIR/airline_A_baseline.json"

# ── Phase B: Lyapunov-only ───────────────────────────────────
run_phase "Phase B: Lyapunov-only (no RG, no VSA)" \
    "harness_agent" "$RESULTS_DIR/airline_B_lyapunov.json" \
    HARNESS_RG=off HARNESS_VSA=off

# ── Phase C: Lyapunov+RG ────────────────────────────────────
run_phase "Phase C: Lyapunov+RG (no VSA)" \
    "harness_agent" "$RESULTS_DIR/airline_C_lyapunov_rg.json" \
    HARNESS_VSA=off

# ── Phase D: Full-stack ──────────────────────────────────────
run_phase "Phase D: Full-stack (Lyapunov + RG + VSA)" \
    "harness_agent" "$RESULTS_DIR/airline_D_fullstack.json"

# ── Phase E: Naive Cap ───────────────────────────────────────
run_phase "Phase E: Naive Cap (hard 100K budget)" \
    "naive_cap_agent" "$RESULTS_DIR/airline_E_naive_cap.json"

# ── Summary ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  τ³-bench AIRLINE 5-PHASE COMPLETE"                        | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"               | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# Quick infra check
echo "" | tee -a "$LOG_FILE"
echo "Infra error check:" | tee -a "$LOG_FILE"
python3 -c "
import json, os, glob
for f in sorted(glob.glob('$RESULTS_DIR/airline_*.json')):
    with open(f) as fh:
        sims = json.load(fh)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    passed = sum(1 for s in sims if s.get('reward', 0) > 0)
    label = os.path.basename(f)
    icon = '✅' if infra == 0 else '❌'
    print(f'  {icon} {label}: {passed}/{len(sims)} passed, {infra} infra errors')
" 2>&1 | tee -a "$LOG_FILE"
