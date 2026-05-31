#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  State-Harness Full Benchmark Suite
#  Run overnight: ./benchmarks/run_full_benchmark.sh
#
#  What it does:
#    1. Runs BASELINE (llm_agent) on all 4 domains × 3 trials
#    2. Runs GUARDED (harness_agent) on all 4 domains × 3 trials
#    3. Generates per-domain comparison reports
#    4. Generates a combined summary report
#
#  Estimated: ~6.5 hours, ~$46 (₹3,800) on Gemini 2.5 Flash
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAU3_DIR="$(cd "$BENCH_DIR/../tau3-bench" && pwd)"
RESULTS_DIR="$BENCH_DIR/benchmark_results/$(date +%Y%m%d_%H%M%S)"

# Using Vertex AI (no daily quota caps, pay-per-use).
# Requires GOOGLE_APPLICATION_CREDENTIALS or gcloud auth.
AGENT_LLM="vertex_ai/gemini-2.5-flash"
USER_LLM="vertex_ai/gemini-2.5-flash"
NUM_TRIALS=3
SEED=42
MAX_CONCURRENCY=1
MAX_RETRIES=3

# Domains to benchmark (all meaningful τ³-bench domains)
# banking_knowledge excluded — requires OpenAI embeddings (text-embedding-3-large)
DOMAINS=("airline" "retail" "telecom")

# Agents to compare (3-way: no guard vs. naive cap vs. smart guard)
AGENTS=("llm_agent" "naive_cap_agent" "harness_agent")

# ── Setup ──────────────────────────────────────────────────────────────

# Initialize pyenv (required for nohup/non-interactive shells)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

# Load credentials from .env (safe extraction — avoids sourcing broken placeholders)
if [ -f "$TAU3_DIR/.env" ]; then
    # Vertex AI auth: service account JSON key
    VERTEX_PROJECT=$(grep '^VERTEXAI_PROJECT=' "$TAU3_DIR/.env" | cut -d'=' -f2-)
    VERTEX_LOCATION=$(grep '^VERTEXAI_LOCATION=' "$TAU3_DIR/.env" | cut -d'=' -f2-)
    GOOGLE_CREDS=$(grep '^GOOGLE_APPLICATION_CREDENTIALS=' "$TAU3_DIR/.env" | cut -d'=' -f2-)

    [ -n "$VERTEX_PROJECT" ] && export VERTEXAI_PROJECT="$VERTEX_PROJECT"
    [ -n "$VERTEX_LOCATION" ] && export VERTEXAI_LOCATION="$VERTEX_LOCATION"
    [ -n "$GOOGLE_CREDS" ] && export GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_CREDS"

    # Fallback: also load GEMINI_API_KEY if present (for gemini/ prefix)
    GEMINI_KEY=$(grep '^GEMINI_API_KEY=' "$TAU3_DIR/.env" | cut -d'=' -f2-)
    [ -n "$GEMINI_KEY" ] && export GEMINI_API_KEY="$GEMINI_KEY"
fi

mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/benchmark.log"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# Activate τ³-bench venv
source "$TAU3_DIR/.venv/bin/activate"

log "═══════════════════════════════════════════════════════════════"
log "  State-Harness Full Benchmark Suite"
log "═══════════════════════════════════════════════════════════════"
log "  Results directory: $RESULTS_DIR"
log "  Agent LLM: $AGENT_LLM"
log "  User LLM:  $USER_LLM"
log "  Trials:    $NUM_TRIALS"
log "  Domains:   ${DOMAINS[*]}"
log "  Agents:    ${AGENTS[*]}"
log "═══════════════════════════════════════════════════════════════"
log ""

# Track timing
BENCH_START=$(date +%s)

# ── Run Benchmarks ─────────────────────────────────────────────────────

run_count=0
total_runs=$(( ${#DOMAINS[@]} * ${#AGENTS[@]} ))

for domain in "${DOMAINS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        run_count=$((run_count + 1))
        run_name="${domain}_${agent}"
        run_dir="$RESULTS_DIR/$run_name"
        mkdir -p "$run_dir"

        log "────────────────────────────────────────────────────────"
        log "  [$run_count/$total_runs] Domain: $domain | Agent: $agent"
        log "────────────────────────────────────────────────────────"

        run_start=$(date +%s)

        # Run τ³-bench (tau2 is in the activated venv, no uv needed)
        tau2 run \
            --domain "$domain" \
            --agent "$agent" \
            --agent-llm "$AGENT_LLM" \
            --user-llm "$USER_LLM" \
            --num-trials "$NUM_TRIALS" \
            --seed "$SEED" \
            --max-concurrency "$MAX_CONCURRENCY" \
            --max-retries "$MAX_RETRIES" \
            --save-to "$run_dir/results.json" \
            --log-level WARNING \
            2>&1 | tee -a "$LOG_FILE"

        run_end=$(date +%s)
        run_duration=$(( run_end - run_start ))
        log "  ✅ Completed in ${run_duration}s"
        log ""

        # Find the actual results.json (τ³-bench nests it)
        actual_results=$(find "$run_dir" -name "results.json" -type f | head -1)
        if [ -n "$actual_results" ]; then
            # Copy to a predictable location
            cp "$actual_results" "$RESULTS_DIR/${run_name}.json"
            log "  📄 Results saved: ${run_name}.json"
        else
            log "  ⚠️  No results.json found for $run_name"
        fi
    done
done

# ── Generate Analysis Reports ─────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════════════════"
log "  Generating Analysis Reports"
log "═══════════════════════════════════════════════════════════════"

ANALYZE_SCRIPT="$BENCH_DIR/benchmarks/analyze_results.py"

{
    echo "═══════════════════════════════════════════════════════════════"
    echo "  STATE-HARNESS FULL BENCHMARK REPORT"
    echo "  Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Agent LLM: $AGENT_LLM"
    echo "  Trials: $NUM_TRIALS per task"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
} > "$SUMMARY_FILE"

for domain in "${DOMAINS[@]}"; do
    baseline_file="$RESULTS_DIR/${domain}_llm_agent.json"
    guarded_file="$RESULTS_DIR/${domain}_harness_agent.json"

    if [ -f "$baseline_file" ] && [ -f "$guarded_file" ]; then
        log "  Analyzing: $domain"
        python3 "$ANALYZE_SCRIPT" \
            --baseline "$baseline_file" \
            --guarded "$guarded_file" \
            2>&1 | tee -a "$SUMMARY_FILE"
    else
        log "  ⚠️  Missing results for $domain (baseline: $([ -f "$baseline_file" ] && echo "✓" || echo "✗"), guarded: $([ -f "$guarded_file" ] && echo "✓" || echo "✗"))"
        echo "  ⚠️  Missing results for $domain" >> "$SUMMARY_FILE"
    fi
done

# ── Final Summary ─────────────────────────────────────────────────────

BENCH_END=$(date +%s)
BENCH_DURATION=$(( BENCH_END - BENCH_START ))
BENCH_HOURS=$(( BENCH_DURATION / 3600 ))
BENCH_MINS=$(( (BENCH_DURATION % 3600) / 60 ))

{
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  BENCHMARK COMPLETE"
    echo "  Total time: ${BENCH_HOURS}h ${BENCH_MINS}m"
    echo "  Results: $RESULTS_DIR"
    echo "═══════════════════════════════════════════════════════════════"
} | tee -a "$SUMMARY_FILE" | tee -a "$LOG_FILE"

log ""
log "Full report saved to: $SUMMARY_FILE"
log "To view: cat $SUMMARY_FILE"
