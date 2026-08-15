#!/usr/bin/env bash
# =============================================================================
# SWE-bench Benchmark Setup for state-harness evaluation
# =============================================================================
#
# This script sets up the moatless-tools environment for running SWE-bench
# with and without state-harness monitoring.
#
# Prerequisites:
#   - Python 3.12+ (3.13 recommended)
#   - Docker or OrbStack running
#   - Vertex AI credentials (service account JSON)
#   - Voyage AI API key (free at https://dash.voyageai.com/)
#
# Usage:
#   ./benchmarks/swe_bench/setup.sh
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$(cd "$STATE_HARNESS_DIR/.." && pwd)"
MOATLESS_DIR="$PARENT_DIR/moatless-tools"

echo "================================================"
echo " SWE-bench Setup for state-harness Evaluation"
echo "================================================"
echo ""
echo "State-harness: $STATE_HARNESS_DIR"
echo "Moatless-tools: $MOATLESS_DIR"
echo ""

# ── Step 1: Clone moatless-tools ──────────────────────────────────────────────
if [ -d "$MOATLESS_DIR" ]; then
    echo "✅ moatless-tools already exists at $MOATLESS_DIR"
else
    echo "📦 Cloning moatless-tools..."
    git clone https://github.com/aorwall/moatless-tools.git "$MOATLESS_DIR"
fi

cd "$MOATLESS_DIR"

# ── Step 2: Create virtual environment ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "✅ Virtual environment activated"

# ── Step 3: Install dependencies ──────────────────────────────────────────────
echo "📦 Installing moatless-tools dependencies..."
pip install -e ".[all]" --quiet

# Pin litellm to avoid known compatibility issues
pip install "litellm>=1.75.0,<1.80.0" --quiet

echo "✅ moatless-tools installed"

# ── Step 4: Install state-harness ─────────────────────────────────────────────
echo "📦 Installing state-harness..."
pip install -e "$STATE_HARNESS_DIR" --quiet
echo "✅ state-harness installed"

# Verify imports
python3 -c "from state_harness import GrowthRatioGuard; print('✅ state-harness import OK')"
python3 -c "from moatless.flow.search_tree import SearchTree; print('✅ moatless import OK')"

# ── Step 5: Apply state-harness patches ───────────────────────────────────────
echo "🔧 Applying state-harness integration patches..."

# Copy harness_loop.py (HarnessSearchTree + HarnessLoop)
cp "$SCRIPT_DIR/harness_loop.py" "$MOATLESS_DIR/moatless/flow/harness_loop.py"
echo "   → Installed harness_loop.py"

# Apply docker_run.py patch (model arg fix + Vertex AI credential forwarding)
if [ -f "$SCRIPT_DIR/docker_run.patch" ]; then
    cd "$MOATLESS_DIR"
    git apply "$SCRIPT_DIR/docker_run.patch" 2>/dev/null || echo "   ⚠️ Patch already applied or conflicts"
    echo "   → Applied docker_run.patch"
fi

# Copy flow configs
mkdir -p "$MOATLESS_DIR/.moatless/flows"
cp "$SCRIPT_DIR/flow_configs/swebench_harness.json" "$MOATLESS_DIR/.moatless/flows/"
cp "$SCRIPT_DIR/flow_configs/swebench_baseline.json" "$MOATLESS_DIR/.moatless/flows/"
echo "   → Installed flow configs"

echo "✅ Patches applied"

# ── Step 6: Docker network ────────────────────────────────────────────────────
if docker network inspect moatless-network &>/dev/null; then
    echo "✅ Docker network 'moatless-network' exists"
else
    echo "🐳 Creating Docker network..."
    docker network create moatless-network
    echo "✅ Docker network created"
fi

# ── Step 7: Verify environment ────────────────────────────────────────────────
echo ""
echo "================================================"
echo " Verification"
echo "================================================"

# Check required env vars
MISSING=()
[ -z "${VOYAGE_API_KEY:-}" ] && MISSING+=("VOYAGE_API_KEY")
[ -z "${VERTEXAI_PROJECT:-}" ] && MISSING+=("VERTEXAI_PROJECT")
[ -z "${VERTEXAI_LOCATION:-}" ] && MISSING+=("VERTEXAI_LOCATION")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing environment variables:"
    for var in "${MISSING[@]}"; do
        echo "   export $var=<your-value>"
    done
    echo ""
    echo "Set them in your shell or in a .env file before running benchmarks."
else
    echo "✅ All environment variables set"
fi

docker info &>/dev/null && echo "✅ Docker is running" || echo "❌ Docker is NOT running"

echo ""
echo "================================================"
echo " Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Set environment variables (see above)"
echo "  2. Run a test: ./benchmarks/swe_bench/run_single_test.sh"
echo "  3. Run full benchmark: ./benchmarks/swe_bench/run_benchmark.sh"
echo ""
