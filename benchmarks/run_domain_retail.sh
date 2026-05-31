#!/bin/bash
# Retail domain runner (5 phases incl. threshold sweep, max-concurrency=5)
set -euo pipefail

TAU3_DIR="$(dirname "$PROJECT_ROOT")/tau3-bench"
cd "$TAU3_DIR"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
CONC=5

echo "[$(date '+%H:%M:%S')] ═══ RETAIL START (5 phases incl. sweep) ═══"

echo "[$(date '+%H:%M:%S')] Phase 1/5: Retail Baseline"
uv run tau2 run --domain retail --agent llm_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/retail_llm_agent.json"

echo "[$(date '+%H:%M:%S')] Phase 2/5: Retail Naive Cap"
uv run tau2 run --domain retail --agent naive_cap_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/retail_naive_cap_agent.json"

echo "[$(date '+%H:%M:%S')] Phase 3/5: Retail Harness τ=2.0 (sweep)"
HARNESS_RATIO_THRESHOLD=2.0 HARNESS_BUDGET_GATE=8000 \
uv run tau2 run --domain retail --agent harness_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/retail_harness_tau2.0.json"

echo "[$(date '+%H:%M:%S')] Phase 4/5: Retail Harness τ=2.5 (main)"
uv run tau2 run --domain retail --agent harness_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/retail_harness_tau2.5.json"
# No env override — uses DOMAIN_THRESHOLDS retail default (τ=2.5, gate=12000)

echo "[$(date '+%H:%M:%S')] Phase 5/5: Retail Harness τ=3.0 (sweep)"
HARNESS_RATIO_THRESHOLD=3.0 HARNESS_BUDGET_GATE=12000 \
uv run tau2 run --domain retail --agent harness_agent \
    --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" \
    --num-trials 3 --max-concurrency $CONC \
    --save-to "${RESULTS_DIR}/retail_harness_tau3.0.json"

echo "[$(date '+%H:%M:%S')] ═══ RETAIL COMPLETE ═══"

# Quick infra check
python3 -c "
import json, os
for f in ['retail_llm_agent.json','retail_naive_cap_agent.json',
          'retail_harness_tau2.0.json','retail_harness_tau2.5.json','retail_harness_tau3.0.json']:
    path = os.path.join('$RESULTS_DIR', f)
    if not os.path.exists(path): continue
    with open(path) as fh:
        sims = json.load(fh)['simulations']
    infra = sum(1 for s in sims if s.get('termination_reason') == 'infrastructure_error')
    print(f'  {\"✅\" if infra == 0 else \"❌\"} {f}: {infra}/{len(sims)} infra errors')
"
