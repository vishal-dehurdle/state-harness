#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Local Model Setup — Pull required Ollama models for benchmarking
#
# Models (optimized for 16GB MacBook):
#   - qwen3:4b    (2.5 GB) — Alibaba's edge model
#   - llama3.2:3b (2.0 GB) — Meta's compact model
#   - phi4-mini   (2.5 GB) — Microsoft's reasoning model
#   - gemma4:e4b  (9.6 GB) — Already downloaded (borderline edge)
#
# Total download: ~7 GB for new models
# ════════════════════════════════════════════════════════════════

set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "  Ollama Model Setup for Local Benchmarks"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not installed. Install from https://ollama.com"
    exit 1
fi

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. Starting..."
    ollama serve &
    sleep 3
fi

echo "Current models:"
ollama list
echo ""

# Pull models
MODELS=("qwen3:4b" "llama3.2:3b" "phi4-mini")

for model in "${MODELS[@]}"; do
    echo "──────────────────────────────────────────────────────────"
    echo "  Pulling $model..."
    echo "──────────────────────────────────────────────────────────"
    ollama pull "$model"
    echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Models available:"
ollama list
echo ""
echo "Run benchmarks with:"
echo "  python benchmarks/local_models/run_local_benchmark.py --model qwen3:4b"
echo "  python benchmarks/local_models/run_local_benchmark.py --model llama3.2:3b"
echo "  python benchmarks/local_models/run_local_benchmark.py --model phi4-mini"
echo "  python benchmarks/local_models/run_local_benchmark.py --model gemma4:e4b"
