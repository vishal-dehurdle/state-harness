"""
MINT Harness Runner — runs MINT benchmark with state-harness monitoring.

This replaces the standard interactive_loop with harness_interactive_loop
which wraps each agent turn with GrowthRatioGuard monitoring.

Usage:
    python3 run_harness_mint.py --exp_config CONFIG.json [--debug]
"""

import sys
import os
import json
import pathlib
import importlib
import argparse
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Pure Python fallback — avoids requiring Rust extension build
# This implements the same GrowthRatioGuard algorithm as the Rust core
import statistics

# Add mint-bench to path
MINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "mint-bench")
sys.path.insert(0, MINT_DIR)

from mint.envs import GeneralEnv
from mint.datatypes import Action, State
from mint.tools import Tool
import mint.tasks as tasks
import mint.agents as agents

class StabilityViolation(Exception):
    """Growth ratio exceeded threshold for W consecutive steps."""
    pass

class BudgetExhausted(Exception):
    """Total token budget exceeded."""
    pass

class GrowthRatioGuard:
    """Pure Python GrowthRatioGuard — identical algorithm to Rust core.
    
    Monitors per-turn token consumption. After a warmup period, computes
    growth ratio R(k) = S(k) / baseline. Trips when R(k) > threshold
    for `window` consecutive steps AND total tokens > budget_gate.
    """
    def __init__(self, budget: int = 200_000, window: int = 3,
                 threshold: float = 2.0, warmup: int = 3, budget_gate: int = 8000):
        self.budget = budget
        self.window = window
        self.threshold = threshold
        self.warmup = warmup
        self.budget_gate = budget_gate
        self._steps: list[int] = []
        self._total_tokens = 0
        self._consecutive_violations = 0
        self._baseline: float | None = None
        self.tripped = False
        self.trip_reason: str | None = None

    def record_step(self, tokens_used: int):
        """Record one step's token usage. Raises on violation."""
        self._steps.append(tokens_used)
        self._total_tokens += tokens_used

        # Budget check
        if self._total_tokens > self.budget:
            self.tripped = True
            self.trip_reason = "budget_exhausted"
            raise BudgetExhausted(
                f"Total tokens {self._total_tokens} exceeded budget {self.budget}"
            )

        # Establish baseline after warmup
        if len(self._steps) == self.warmup:
            self._baseline = statistics.median(self._steps)
            return

        # Not enough data yet
        if self._baseline is None or self._baseline == 0:
            return

        # Compute growth ratio
        ratio = tokens_used / self._baseline

        if ratio > self.threshold and self._total_tokens > self.budget_gate:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        if self._consecutive_violations >= self.window:
            self.tripped = True
            self.trip_reason = "stability_violation"
            raise StabilityViolation(
                f"Growth ratio {ratio:.2f}x exceeded {self.threshold}x "
                f"for {self.window} consecutive steps"
            )

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("MINT.Harness")


def harness_interactive_loop(
    task: tasks.Task,
    agent: agents.LMAgent,
    tools: List[Tool],
    feedback_config: Dict[str, Any],
    env_config: Dict[str, Any],
    budget: int = 200_000,
    window: int = 3,
    threshold: float = 2.0,
) -> State:
    """MINT interactive loop with state-harness Lyapunov monitoring."""

    env = GeneralEnv(task, tools, feedback_config, env_config)
    state: State = env.reset()

    guard = GrowthRatioGuard(budget=budget, window=window, threshold=threshold)
    LOGGER.debug(f"Harness: budget={budget}, window={window}, threshold={threshold}")

    num_steps = 0
    harness_terminated = False
    prev_total_tokens = 0

    # Load history if any
    if task.loaded_history is not None:
        for turn in task.loaded_history:
            action = agent.lm_output_to_action(turn["lm_output"])
            state = env.step(action, loaded=turn)
            num_steps += 1

    while not state.finished:
        prev_total_tokens = sum(state.token_counter.values())

        # Agent acts
        action: Action = agent.act(state)

        # Check harness after agent turn
        current_total = sum(state.token_counter.values())
        tokens_this_turn = current_total - prev_total_tokens

        if tokens_this_turn > 0:
            try:
                guard.record_step(tokens_used=tokens_this_turn)
            except StabilityViolation as e:
                LOGGER.warning(f"🛑 Stability violation at step {num_steps}: {e}")
                harness_terminated = True
                break
            except BudgetExhausted as e:
                LOGGER.warning(f"🛑 Budget exhausted at step {num_steps}: {e}")
                harness_terminated = True
                break

        # Environment step
        state: State = env.step(action)
        num_steps += 1

    termination = "harness" if harness_terminated else "natural"
    total_tokens = sum(state.token_counter.values())

    LOGGER.info(
        f"Task finished in {num_steps} steps. "
        f"Success: {state.success}. "
        f"Termination: {termination}. "
        f"Tokens: {total_tokens}"
    )

    # Attach harness metadata
    if not hasattr(state, '_harness_meta'):
        state._harness_meta = {}
    state._harness_meta = {
        "terminated_by_harness": harness_terminated,
        "num_steps": num_steps,
        "total_tokens": total_tokens,
        "termination_reason": termination,
    }

    return state


def main(args):
    with open(args.exp_config) as f:
        exp_config: Dict[str, Any] = json.load(f)

    LOGGER.info(f"Harness config: {exp_config}")

    # Modify output dir to indicate harness mode
    original_output = exp_config["output_dir"]
    exp_config["output_dir"] = original_output.replace(
        "gemini-2.5-flash",
        "gemini-2.5-flash-harness"
    )

    # Initialize tasks
    task_config = exp_config["task"]
    task_class = getattr(tasks, task_config["task_class"])
    todo_tasks, n_tasks = task_class.load_tasks(task_config["filepath"])

    # Initialize agent
    agent_config = exp_config["agent"]
    agent = getattr(agents, agent_config["agent_class"])(agent_config["config"])

    # Initialize tools
    tools: List[Tool] = [
        getattr(importlib.import_module(module), class_name)()
        for module, class_name in task_config["tool_imports"]
    ]

    feedback_config = exp_config.get("feedback_config", {})
    env_config = exp_config.get("env_config", {})

    pathlib.Path(exp_config["output_dir"]).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(exp_config["output_dir"], "results.jsonl")

    # Resume support
    done_task_id = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                task_id = json.loads(line)["task"].get("task_id", "")
                if task_id == "":
                    task_id = json.loads(line)["task"].get("id", "")
                done_task_id.add(task_id)
        LOGGER.info(f"Resuming: {len(done_task_id)} tasks already done.")

    if len(done_task_id) == n_tasks:
        LOGGER.info("All tasks done. Exiting.")
        return

    n_remaining = n_tasks - len(done_task_id)
    LOGGER.info(f"Running harness loop for {n_remaining} tasks.")

    with open(output_path, "a") as f, logging_redirect_tqdm():
        pbar = tqdm(total=n_remaining)
        for i, task in enumerate(todo_tasks):
            if args.debug and i == 3:
                break
            if task.task_id in done_task_id:
                continue

            state = harness_interactive_loop(
                task, agent, tools, feedback_config, env_config,
                budget=200_000, window=3, threshold=2.0,
            )

            # Write result with harness metadata
            result = {"state": state.to_dict(), "task": task.to_dict()}
            if hasattr(state, '_harness_meta'):
                result["harness"] = state._harness_meta
            f.write(json.dumps(result) + "\n")
            f.flush()
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run MINT with state-harness monitoring.")
    parser.add_argument("--exp_config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args)
