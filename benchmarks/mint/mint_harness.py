"""
State-harness wrapper for MINT benchmark.

This module wraps MINT's interactive_loop() with GrowthRatioGuard monitoring
to detect and terminate spiraling agent loops.

Usage:
    # Replace the standard interactive_loop call in main.py with:
    from mint_harness import harness_interactive_loop
    state = harness_interactive_loop(task, agent, tools, feedback_config, env_config)
"""

import logging
from typing import List, Dict, Any

from mint.envs import GeneralEnv, AlfworldEnv
from mint.datatypes import Action, State
from mint.tasks import AlfWorldTask
from mint.tools import Tool
import mint.tasks as tasks
import mint.agents as agents

LOGGER = logging.getLogger("MINT.Harness")

# Attempt to import state_harness
try:
    from state_harness import GrowthRatioGuard, StabilityViolation, BudgetExhausted
    HAS_STATE_HARNESS = True
except ImportError:
    HAS_STATE_HARNESS = False
    LOGGER.warning("state_harness not installed. Running without harness monitoring.")


def harness_interactive_loop(
    task: tasks.Task,
    agent: agents.LMAgent,
    tools: List[Tool],
    feedback_config: Dict[str, Any],
    env_config: Dict[str, Any],
    interactive_mode: bool = False,
    # Harness parameters
    budget: int = 200_000,
    window: int = 5,
    threshold: float = 1.15,
) -> State:
    """
    MINT interactive loop with state-harness Lyapunov monitoring.

    Same as MINT's standard interactive_loop but wraps each agent turn
    with GrowthRatioGuard to detect token consumption spirals.

    Args:
        task, agent, tools, feedback_config, env_config, interactive_mode:
            Standard MINT parameters (see mint.main.interactive_loop)
        budget: Total token budget for the guard
        window: Rolling window size for growth ratio computation
        threshold: Growth ratio threshold (>1.0 means growing)
    """
    if isinstance(task, AlfWorldTask):
        env = AlfworldEnv(task, tools, feedback_config, env_config)
    else:
        env = GeneralEnv(task, tools, feedback_config, env_config)

    state: State = env.reset()

    LOGGER.info(f"\nUser: \n\033[94m{state.latest_output['content']}\033[0m")

    # Initialize harness guard
    guard = None
    if HAS_STATE_HARNESS:
        guard = GrowthRatioGuard(budget=budget, window=window, threshold=threshold)
        LOGGER.info(f"Harness guard initialized: budget={budget}, window={window}, threshold={threshold}")

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
        if interactive_mode:
            to_continue = "n"
            while to_continue not in ["y", "Y"]:
                to_continue = input("\n> Continue? (y/n) ")

        # Record tokens before this turn
        prev_total_tokens = sum(state.token_counter.values())

        # Agent acts
        action: Action = agent.act(state)

        LOGGER.info(
            f"\n\033[1m" + "LM Agent Action:\n" + "\033[0m" +
            f"\n\033[92m{action.value}\033[0m"
        )

        # Check harness after agent's turn
        if guard is not None:
            current_total = sum(state.token_counter.values())
            tokens_this_turn = current_total - prev_total_tokens

            if tokens_this_turn > 0:
                try:
                    guard.record_step(tokens_used=tokens_this_turn)
                    ratio = guard.current_ratio()
                    LOGGER.debug(f"Harness step {num_steps}: tokens={tokens_this_turn}, ratio={ratio:.3f}")
                except StabilityViolation as e:
                    LOGGER.warning(f"🛑 Harness: Stability violation at step {num_steps}: {e}")
                    harness_terminated = True
                    break
                except BudgetExhausted as e:
                    LOGGER.warning(f"🛑 Harness: Budget exhausted at step {num_steps}: {e}")
                    harness_terminated = True
                    break

        # Environment step
        state: State = env.step(action)

        if not state.finished:
            user_msg = state.latest_output['content']
            LOGGER.info(
                "\n" + "\033[1m" + "User:\n" + "\033[0m" +
                f"\033[94m{user_msg}\033[0m" + "\n"
            )
        num_steps += 1

    termination_reason = "harness_terminated" if harness_terminated else "natural"
    LOGGER.info(
        f"Task finished in {num_steps} steps. "
        f"Success: {state.success}. "
        f"Termination: {termination_reason}. "
        f"Total tokens: {sum(state.token_counter.values())}"
    )

    # Add harness metadata to state for analysis
    state.harness_metadata = {
        "terminated_by_harness": harness_terminated,
        "num_steps": num_steps,
        "total_tokens": sum(state.token_counter.values()),
        "termination_reason": termination_reason,
    }

    return state
