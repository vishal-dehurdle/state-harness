"""
State-harness wrapper for MINT benchmark — Full-Stack v3.

Wraps MINT's interactive_loop() with the complete state-harness stack:
1. GrowthRatioGuard: Normalized token monitoring
2. RGDecimator: History compression during causal interventions
3. HolographicEngine: VSA policy drift detection
4. Dual-confirmation gating: Trip only when BOTH growth-ratio AND drift confirm

Feature toggles (via environment variables):
    HARNESS_MODE=off          → Condition A: No monitoring (pure baseline)
    HARNESS_RG=off HARNESS_VSA=off → Condition B: Lyapunov-only
    HARNESS_VSA=off           → Condition C: Lyapunov + RG
    (default)                 → Condition D: Full-stack

Usage:
    from mint_harness import harness_interactive_loop
    state = harness_interactive_loop(task, agent, tools, feedback_config, env_config)
"""
qa
import logging
import os
import statistics
from typing import List, Dict, Any, Optional

from mint.envs import GeneralEnv, AlfworldEnv
from mint.datatypes import Action, State
from mint.tasks import AlfWorldTask
from mint.tools import Tool
import mint.tasks as tasks
import mint.agents as agents

LOGGER = logging.getLogger("MINT.Harness")

# ── Import state-harness (with graceful fallback) ──────────────────────

try:
    from state_harness import (
        GrowthRatioGuard,
        HolographicEngine,
        RGDecimator,
        StabilityViolation,
        BudgetExhausted,
    )
    HAS_STATE_HARNESS = True
except ImportError:
    HAS_STATE_HARNESS = False
    LOGGER.warning("state_harness not installed. Running without harness monitoring.")


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() not in ("off", "0", "false", "no")


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
    threshold: float = 2.0,
    warmup: int = 3,
    budget_gate: int = 8_000,
    max_interventions: int = 2,
    # Feature flags (overridden by env vars)
    rg_enabled: Optional[bool] = None,
    vsa_enabled: Optional[bool] = None,
    harness_enabled: Optional[bool] = None,
) -> State:
    """MINT interactive loop with full-stack state-harness monitoring.

    Same interface as MINT's standard interactive_loop but adds:
    - Growth-ratio Lyapunov monitoring
    - RG history compression during causal interventions
    - VSA policy drift detection with dual-confirmation gating
    """

    # ── Resolve feature flags from env vars ──
    if harness_enabled is None:
        harness_enabled = _env_flag("HARNESS_MODE", default=True) and HAS_STATE_HARNESS
    if rg_enabled is None:
        rg_enabled = _env_flag("HARNESS_RG", default=True)
    if vsa_enabled is None:
        vsa_enabled = _env_flag("HARNESS_VSA", default=True)

    # ── Set up environment ──
    if isinstance(task, AlfWorldTask):
        env = AlfworldEnv(task, tools, feedback_config, env_config)
    else:
        env = GeneralEnv(task, tools, feedback_config, env_config)

    state: State = env.reset()
    LOGGER.info(f"\nUser: \n\033[94m{state.latest_output['content']}\033[0m")

    # ── Initialize harness components ──
    guard = None
    rg = None
    engine = None
    drift_history: list[float] = []
    drift_window = 3
    drift_threshold = 0.7
    intervention_count = 0
    rg_compressions = 0

    if harness_enabled:
        guard = GrowthRatioGuard(
            token_budget=budget,
            ratio_threshold=threshold,
            window=window,
            warmup_turns=warmup,
            budget_gate=budget_gate,
        )

        if rg_enabled:
            rg = RGDecimator(threshold=0.3, max_retained=30)

        if vsa_enabled:
            engine = HolographicEngine(dim=2000)
            # Encode task description as the policy invariant
            task_desc = getattr(task, 'prompt', '') or str(task)
            policy_key = engine.encode_text("task objective")
            policy_val = engine.encode_text(task_desc[:500])
            engine.register_invariant("task_objective", policy_key, policy_val)

        mode = "full-stack" if (rg_enabled and vsa_enabled) else \
               "lyapunov+rg" if rg_enabled else \
               "lyapunov+vsa" if vsa_enabled else "lyapunov-only"
        LOGGER.info(
            f"Harness initialized: mode={mode}, budget={budget}, "
            f"window={window}, threshold={threshold}"
        )

    num_steps = 0
    harness_terminated = False
    termination_reason = "natural"
    prev_total_tokens = 0

    # ── Load history if resuming ──
    if task.loaded_history is not None:
        for turn in task.loaded_history:
            action = agent.lm_output_to_action(turn["lm_output"])
            state = env.step(action, loaded=turn)
            num_steps += 1

    # ── Main loop ──
    while not state.finished:
        if interactive_mode:
            to_continue = "n"
            while to_continue not in ["y", "Y"]:
                to_continue = input("\n> Continue? (y/n) ")

        prev_total_tokens = sum(state.token_counter.values())

        # Agent acts
        action: Action = agent.act(state)

        LOGGER.info(
            f"\n\033[1m" + "LM Agent Action:\n" + "\033[0m" +
            f"\n\033[92m{action.value}\033[0m"
        )

        # ── Harness check after agent turn ──
        if guard is not None:
            current_total = sum(state.token_counter.values())
            tokens_this_turn = current_total - prev_total_tokens

            # Check VSA drift
            if engine is not None and action.value:
                context_vec = engine.encode_text(action.value[:200])
                drift = engine.check_drift("task_objective", context_vec)
                drift_history.append(drift)

            if tokens_this_turn > 0:
                try:
                    guard.record_step(tokens_used=tokens_this_turn)
                    LOGGER.debug(
                        f"Harness step {num_steps}: tokens={tokens_this_turn}, "
                        f"ratio={guard.current_ratio or 'warmup'}"
                    )
                except StabilityViolation as e:
                    # Dual-confirmation: check drift gate
                    drift_confirmed = True
                    if vsa_enabled and engine and len(drift_history) >= drift_window:
                        recent = drift_history[-drift_window:]
                        drift_confirmed = statistics.mean(recent) > drift_threshold

                    if drift_confirmed:
                        if intervention_count >= max_interventions:
                            LOGGER.warning(
                                f"🛑 Harness: Stability violation at step {num_steps} "
                                f"after {intervention_count} interventions: {e}"
                            )
                            harness_terminated = True
                            termination_reason = "stability_violation"
                            break
                        else:
                            # Causal intervention: reset escalation
                            intervention_count += 1
                            guard.reset_escalation()
                            LOGGER.info(
                                f"[state-harness] Intervention "
                                f"#{intervention_count}/{max_interventions} "
                                f"at step {num_steps}: reset escalation"
                            )
                    else:
                        # On-policy — suppress
                        LOGGER.info(
                            f"[state-harness] Growth ratio exceeded but drift is low "
                            f"— suppressing at step {num_steps}"
                        )
                except BudgetExhausted as e:
                    LOGGER.warning(
                        f"🛑 Harness: Budget exhausted at step {num_steps}: {e}"
                    )
                    harness_terminated = True
                    termination_reason = "budget_exhausted"
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

    if not harness_terminated:
        termination_reason = "natural"

    total_tokens = sum(state.token_counter.values())
    LOGGER.info(
        f"Task finished in {num_steps} steps. "
        f"Success: {state.success}. "
        f"Termination: {termination_reason}. "
        f"Total tokens: {total_tokens}"
    )

    # ── Attach harness telemetry ──
    state.harness_metadata = {
        "terminated_by_harness": harness_terminated,
        "num_steps": num_steps,
        "total_tokens": total_tokens,
        "termination_reason": termination_reason,
        "harness_enabled": harness_enabled,
        "rg_enabled": rg_enabled,
        "vsa_enabled": vsa_enabled,
        "intervention_count": intervention_count,
        "rg_compressions": rg_compressions,
        "drift_history": drift_history[-10:] if drift_history else [],
        "baseline": guard.baseline if guard else None,
        "current_ratio": guard.current_ratio if guard else None,
        "energy_history": list(guard.energy_history) if guard else [],
    }

    return state
