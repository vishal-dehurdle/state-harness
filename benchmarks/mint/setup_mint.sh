#!/usr/bin/env bash
# =============================================================================
# MINT Benchmark Setup for state-harness evaluation
# =============================================================================
# One-time setup: clones mint-bench, installs dependencies, patches imports
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_HARNESS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MINT_DIR="$(cd "$STATE_HARNESS_DIR/.." && pwd)/mint-bench"

echo "=== MINT Benchmark Setup ==="
echo "State-harness: $STATE_HARNESS_DIR"
echo "MINT target:   $MINT_DIR"
echo ""

# Step 1: Clone mint-bench if needed
if [ ! -d "$MINT_DIR" ]; then
    echo "📦 Cloning mint-bench..."
    git clone https://github.com/xingyaoww/mint-bench.git "$MINT_DIR"
else
    echo "✅ mint-bench already cloned"
fi

# Step 2: Create virtualenv
cd "$MINT_DIR"
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtualenv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Step 3: Install core deps (skip heavy alfworld)
echo "📥 Installing dependencies..."
pip install litellm backoff tqdm sympy wikipedia langchain langchain-community langchain-core ipython pyyaml google-cloud-aiplatform --quiet

# Step 4: Install state-harness
echo "📥 Installing state-harness..."
pip install -e "$STATE_HARNESS_DIR" --quiet 2>/dev/null || pip install "$STATE_HARNESS_DIR" --quiet

# Step 5: Patch imports (make alfworld optional)
echo "🔧 Patching imports..."

# Patch tasks/__init__.py
if grep -q "^from .alfworld import" mint/tasks/__init__.py 2>/dev/null; then
    cat > mint/tasks/__init__.py << 'PYEOF'
from .base import Task
from .reasoning import ReasoningTask, MultipleChoiceTask, TheoremqaTask
from .codegen import CodeGenTask, HumanEvalTask, MBPPTask

try:
    from .alfworld import AlfWorldTask
except ImportError:
    AlfWorldTask = None
PYEOF
fi

# Patch envs/__init__.py
if grep -q "^from .alfworld_env import" mint/envs/__init__.py 2>/dev/null; then
    cat > mint/envs/__init__.py << 'PYEOF'
from .base import BaseEnv
from .general_env import GeneralEnv

try:
    from .alfworld_env import AlfworldEnv
except ImportError:
    AlfworldEnv = None
PYEOF
fi

# Patch agents/__init__.py (make old OpenAI agents optional, add Gemini)
cat > mint/agents/__init__.py << 'PYEOF'
from .base import LMAgent

try:
    from .openai_lm_agent import OpenAILMAgent
    from .openai_feedback_agent import OpenAIFeedbackAgent
except (AttributeError, ImportError):
    OpenAILMAgent = None
    OpenAIFeedbackAgent = None

try:
    from .bard_agent import BardLMAgent
except (AttributeError, ImportError):
    BardLMAgent = None

try:
    from .claude_feedback_agent import ClaudeFeedbackAgent
    from .claude_agent import ClaudeLMAgent
except (AttributeError, ImportError):
    ClaudeFeedbackAgent = None
    ClaudeLMAgent = None

try:
    from .vllm_feedback_agent import VLLMFeedbackAgent
    from .vllm_agent import VLLMAgent
except (AttributeError, ImportError):
    VLLMFeedbackAgent = None
    VLLMAgent = None

from .gemini_agent import GeminiLMAgent
from .gemini_feedback_agent import GeminiFeedbackAgent
PYEOF

# Step 6: Copy Gemini agent files
echo "📋 Installing Gemini agents..."
cp "$SCRIPT_DIR/gemini_agent.py" mint/agents/gemini_agent.py 2>/dev/null || true
cp "$SCRIPT_DIR/gemini_feedback_agent.py" mint/agents/gemini_feedback_agent.py 2>/dev/null || true

# If agent files don't exist in script dir, they should already be in mint-bench
if [ ! -f "mint/agents/gemini_agent.py" ]; then
    echo "⚠️  Warning: gemini_agent.py not found. Please check benchmarks/mint/ directory."
fi

# Step 7: Verify
echo ""
echo "🔍 Verifying installation..."
python3 -c "from mint.agents import GeminiLMAgent; print('  ✅ GeminiLMAgent OK')"
python3 -c "from mint.envs import GeneralEnv; print('  ✅ GeneralEnv OK')"
python3 -c "from mint.tasks import ReasoningTask; print('  ✅ ReasoningTask OK')"
python3 -c "import litellm; print('  ✅ litellm OK')"

echo ""
echo "✅ MINT setup complete!"
echo ""
echo "Next: Run the benchmark with:"
echo "  cd $SCRIPT_DIR && ./run_mint.sh --mode both"
