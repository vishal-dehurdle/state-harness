#!/bin/bash
# Airline domain runner (3 phases, max-concurrency=5)
set -euo pipefail

TAU3_DIR="$(dirname "$PROJECT_ROOT")/tau3-bench"
cd "$TAU3_DIR"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
CONC=5

echo "[$(date '+%H:%M:%S')] ═══ AIRLINE START ═══"

echo "[$(date '+%H:%M:%S')] Phase 1/3: Airline Baseline"
uv run tau2 run --domain airline --agent llm_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/airline_llm_agent.json"

echo "[$(date '+%H:%M:%S')] Phase 2/3: Airline Naive Cap"
uv run tau2 run --domain airline --agent naive_cap_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/airline_naive_cap_agent.json"

echo "[$(date '+%H:%M:%S')] Phase 3/3: Airline Harness (τ=2.0)"
uv run tau2 run --domain airline --agent harness_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/airline_harness_agent.json"

echo "[$(date '+%H:%M:%S')] ═══ AIRLINE COMPLETE ═══"

# Quick infra check
python3 -c "
import json, os
for f in ['airline_llm_agent.json','airline_naive_cap_agent.json','airline_harness_agent.json']:
    path = os.path.join('$RESULTS_DIR', f)
    if not os.path.exists(path): continue
    with open(path) as fh:
        sims = json.load(fh)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    print(f'  {\"✅\" if infra == 0 else \"❌\"} {f}: {infra}/{len(sims)} infra errors')
"
