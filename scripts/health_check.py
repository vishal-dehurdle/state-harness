#!/usr/bin/env python3
"""Health monitor for benchmark runs. Checks results every N minutes."""
import json
import sys
import os
from pathlib import Path

def check_results(results_dir: str | os.PathLike, label: str = ""):
    """Check a benchmark results directory for errors."""
    results_dir: Path = Path(results_dir)
    if not results_dir.exists():
        print(f"  [{label}] Dir not found: {results_dir}")
        return
    
    for json_dir in sorted(results_dir.iterdir()):
        if not json_dir.is_dir():
            continue
        rf = json_dir / "results.json"
        if not rf.exists():
            print(f"  [{json_dir.name}] No results.json yet")
            continue
        
        try:
            with open(rf) as f:
                data = json.load(f)
            sims = data.get("simulations", [])
            
            passed = sum(1 for s in sims if (s.get("reward_info") or {}).get("reward", 0) > 0)
            infra = sum(1 for s in sims if s.get("termination_reason") == "infrastructure_error")
            zero_msg = sum(1 for s in sims if len(s.get("messages", [])) == 0)
            
            tokens = []
            for s in sims:
                total_t = 0
                for m in s.get("messages", []):
                    u = m.get("usage") or {}
                    total_t += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                tokens.append(total_t)
            avg_tokens = sum(tokens) / max(len(tokens), 1)
            
            status = "✅" if infra == 0 and zero_msg == 0 else "⚠️"
            print(f"  {status} [{json_dir.name}] {len(sims)} runs | "
                  f"pass={passed}/{len(sims)} ({passed*100//max(len(sims),1)}%) | "
                  f"infra_err={infra} | zero_msg={zero_msg} | "
                  f"avg_tokens={avg_tokens:.0f}")
            
            if infra > 0:
                print(f"     🚨 INFRASTRUCTURE ERRORS DETECTED: {infra}")
            if zero_msg > 0:
                print(f"     🚨 ZERO-MESSAGE RUNS DETECTED: {zero_msg}")
                
        except Exception as e:
            print(f"  ❌ [{json_dir.name}] Error reading: {e}")

if __name__ == "__main__":
    from datetime import datetime
    print(f"\n{'='*60}")
    print(f"BENCHMARK HEALTH CHECK — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    base = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "benchmark_results"
    )
    results_path = Path(base)
    if not results_path.exists():
        print(f"\n⚠️  Results directory not found: {base}")
        print("   Usage: python scripts/health_check.py [results_dir]")
        sys.exit(1)

    for d in sorted(results_path.iterdir()):
        if d.is_dir():
            print(f"\n📊 {d.name}:")
            check_results(str(d))

    print(f"\n{'='*60}\n")

