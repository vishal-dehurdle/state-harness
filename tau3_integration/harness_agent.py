"""State-harness wrapped agent for τ³-bench evaluation (v3).

v3 implements the full closed-loop controller from the fluid dynamics paper:

1. **Adaptive Threshold** (§6: adaptive gain scheduling): Auto-calibrates τ
   from warmup variance using Tukey's fence (τ = 1 + k × IQR/median).
   Eliminates manual per-domain threshold tuning.

2. **Causal Intervention** (§4→§6: closed-loop feedback): When instability
   is detected, instead of killing the task, applies aggressive RG compression
   to the conversation history and lets the agent continue with clean context.
   Only kills after max_interventions compressions fail.

3. **Dual-confirmation gating**: Requires BOTH high growth ratio AND high
   policy drift (via HolographicEngine) before acting. If the agent is
   using lots of tokens but staying on-policy, it's doing useful work.

Usage:
    uv run tau2 run --domain airline --agent harness_agent \\
        --agent-llm vertex_ai/gemini-2.5-flash \\
        --user-llm vertex_ai/gemini-2.5-flash
"""

from __future__ import annotations

import statistics
from typing import List, Optional

from loguru import logger

from state_harness import (
    GrowthRatioGuard,
    HolographicEngine,
    RGDecimator,
    StabilityViolation,
    BudgetExhausted,
    PermanentFailure,
)

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMAgentStateType
from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool


# ── Configuration ──────────────────────────────────────────────────────

# Maximum cumulative tokens per task.
DEFAULT_BUDGET_CEILING = 100_000

# Growth ratio threshold: fallback when adaptive calibration has
# insufficient warmup data. With adaptive=True (default), τ is
# auto-calibrated from warmup variance via Tukey's fence.
DEFAULT_RATIO_THRESHOLD = 2.0

# Consecutive escalating turns before circuit breaker trips.
DEFAULT_WINDOW = 3

# Warmup turns to establish baseline token usage.
# 5 turns needed for reliable IQR estimation (adaptive threshold).
DEFAULT_WARMUP_TURNS = 5

# Minimum cumulative tokens before the monitor can trip.
# Prevents killing cheap tasks where even 2x growth is negligible.
DEFAULT_BUDGET_GATE = 8_000

# Lyapunov coupling constant λ.
DEFAULT_LAMBDA = 1.0

# RG decimation settings.
DEFAULT_RG_THRESHOLD = 0.3
DEFAULT_RG_MAX_RETAINED = 30
DEFAULT_RG_ENABLED = True

# VSA dimensionality for policy drift detection.
DEFAULT_VSA_DIM = 2000

# Drift threshold for dual-confirmation gating.
# Only trip if mean recent drift score exceeds this value.
# Higher = more permissive (agent can diverge more before trip).
DEFAULT_DRIFT_THRESHOLD = 0.7

# Adaptive threshold: auto-calibrate τ from warmup variance.
# When True, DOMAIN_THRESHOLDS is not needed — τ self-tunes.
DEFAULT_ADAPTIVE = True
DEFAULT_ADAPTIVE_K = 1.0


class HarnessAgent(LLMAgent[LLMAgentState]):
    """LLMAgent wrapped with state-harness safety monitoring (v2).

    Improvements over v1:
    - Growth-ratio energy function (vs raw token count)
    - RG Decimator compresses history before each LLM call
    - Dual-confirmation: requires both high growth ratio AND high drift
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        budget_ceiling: int = DEFAULT_BUDGET_CEILING,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
        window: int = DEFAULT_WINDOW,
        warmup_turns: int = DEFAULT_WARMUP_TURNS,
        budget_gate: int = DEFAULT_BUDGET_GATE,
        lambda_: float = DEFAULT_LAMBDA,
        rg_threshold: float = DEFAULT_RG_THRESHOLD,
        rg_max_retained: int = DEFAULT_RG_MAX_RETAINED,
        rg_enabled: bool = DEFAULT_RG_ENABLED,
        vsa_enabled: bool = True,
        vsa_dim: int = DEFAULT_VSA_DIM,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
        drift_window: int = 3,
        adaptive: bool = DEFAULT_ADAPTIVE,
        adaptive_k: float = DEFAULT_ADAPTIVE_K,
        max_interventions: int = 2,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

        # ── Growth-ratio Lyapunov monitor ──
        self._guard = GrowthRatioGuard(
            token_budget=budget_ceiling,
            ratio_threshold=ratio_threshold,
            window=window,
            warmup_turns=warmup_turns,
            budget_gate=budget_gate,
            lambda_=lambda_,
            adaptive=adaptive,
            adaptive_k=adaptive_k,
        )

        # ── RG decimator ──
        self._rg = RGDecimator(
            threshold=rg_threshold,
            max_retained=rg_max_retained,
        )
        self._rg_enabled = rg_enabled

        # ── VSA policy drift detector ──
        self._vsa_enabled = vsa_enabled
        if vsa_enabled:
            self._engine = HolographicEngine(dim=vsa_dim)
            self._policy_key = self._engine.encode_text("domain policy compliance")
            self._policy_val = self._engine.encode_text(domain_policy[:500])
            self._engine.register_invariant(
                "policy_compliance", self._policy_key, self._policy_val
            )
        else:
            self._engine = None

        # ── Dual-confirmation config ──
        self._drift_threshold = drift_threshold
        self._drift_window = drift_window

        # ── Causal intervention config ──
        self._max_interventions = max_interventions
        self._intervention_count = 0

        # ── Telemetry accumulators ──
        self._turn_count = 0
        self._total_tokens = 0
        self._drift_history: list[float] = []
        self._rg_compressions: int = 0

        logger.info(
            f"[state-harness] HarnessAgent v3 initialized: "
            f"budget={budget_ceiling}, ratio={ratio_threshold}× "
            f"({'adaptive' if adaptive else 'fixed'}), "
            f"window={window}, warmup={warmup_turns}, "
            f"budget_gate={budget_gate}, "
            f"rg={'on' if rg_enabled else 'off'}, "
            f"vsa={'on' if vsa_enabled else 'off'}, "
            f"drift_threshold={drift_threshold}, "
            f"max_interventions={max_interventions}"
        )

    def _compress_history(self, state: LLMAgentState) -> LLMAgentState:
        """Apply RG decimation to compress conversation history.

        Extracts text from messages, runs the decimator, and replaces
        state.messages with only the retained messages. System messages
        are never touched.
        """
        if not self._rg_enabled or len(state.messages) <= 5:
            return state

        # Extract text content from messages
        texts = []
        for msg in state.messages:
            content = getattr(msg, "content", None) or ""
            if not content:
                # Tool messages may have structured content
                content = str(msg)[:200]
            texts.append(content)

        # Run decimator
        scored = self._rg.decimate(texts)
        retained_indices = {s.index for s in scored if s.retained}

        # Always keep the last 3 messages (recent context is critical)
        n = len(state.messages)
        for i in range(max(0, n - 3), n):
            retained_indices.add(i)

        if len(retained_indices) < len(state.messages):
            original_count = len(state.messages)
            state.messages = [
                state.messages[i]
                for i in sorted(retained_indices)
            ]
            self._rg_compressions += 1
            logger.debug(
                f"[state-harness] RG compressed: {original_count} → "
                f"{len(state.messages)} messages"
            )

        return state

    def _check_drift_gate(self) -> bool:
        """Check if recent drift scores exceed the dual-confirmation threshold.

        Returns True if the agent has drifted enough to confirm a trip.
        Returns True (conservative) if we don't have enough drift data.
        When VSA is disabled, always returns True (Lyapunov-only mode).
        """
        if not self._vsa_enabled:
            return True  # No VSA gate — trip on growth ratio alone

        if len(self._drift_history) < self._drift_window:
            # Not enough data — be conservative and allow trip
            return True

        recent = self._drift_history[-self._drift_window:]
        mean_drift = statistics.mean(recent)
        return mean_drift > self._drift_threshold

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate next message with v3 state-harness monitoring.

        v3 implements closed-loop control (§4→§6 of the fluid dynamics paper):
        detect instability → coarse-grain via RG → continue with clean context.
        Only kills the task after max_interventions compressions fail.
        """

        # ── Step 1: Skip per-turn RG compression ──
        # Per-turn compression was harming healthy tasks by stripping
        # useful context (tool results, earlier conversation). RG is now
        # only applied during causal intervention (when actually needed).

        # ── Step 2: Delegate to base LLMAgent ──
        try:
            assistant_message, state = super().generate_next_message(message, state)
        except Exception as e:
            self._guard.report_transient()
            raise

        # ── Step 3: Extract token usage ──
        tokens_used = 0
        if assistant_message.usage:
            tokens_used = (
                assistant_message.usage.get("completion_tokens", 0)
                + assistant_message.usage.get("prompt_tokens", 0)
            )

        # ── Step 4: Check policy drift via VSA ──
        if self._vsa_enabled and self._engine and assistant_message.content:
            context_vec = self._engine.encode_text(
                assistant_message.content[:200]
            )
            drift = self._engine.check_drift("policy_compliance", context_vec)
            self._drift_history.append(drift)

        # ── Step 5: Record step in growth-ratio monitor ──
        tripped = False
        intervened = False
        try:
            self._guard.record_step(tokens_used=tokens_used, errors=0)
        except StabilityViolation as e:
            # Dual-confirmation: only act if drift also confirms
            # (when VSA is disabled, _check_drift_gate always returns True)
            if self._check_drift_gate():
                if self._intervention_count >= self._max_interventions:
                    # Exhausted interventions — truly stuck, kill it
                    logger.warning(
                        f"[state-harness] Circuit breaker tripped at turn "
                        f"{self._turn_count} after {self._intervention_count} "
                        f"interventions: {e}"
                    )
                    tripped = True
                else:
                    # ── CAUSAL INTERVENTION (§4→§6) ──
                    # Instead of killing, compress context and continue.
                    # This is the closed-loop feedback: detect instability →
                    # coarse-grain → re-inject clean state → continue.
                    self._intervention_count += 1
                    if self._rg_enabled:
                        state = self._apply_causal_intervention(state)
                    self._guard.reset_escalation()
                    intervened = True
                    logger.info(
                        f"[state-harness] Causal intervention "
                        f"#{self._intervention_count}/{self._max_interventions} "
                        f"at turn {self._turn_count}: "
                        f"{'compressed context' if self._rg_enabled else 'reset only'}, "
                        f"reset escalation counter"
                    )
            else:
                # High token growth but agent is on-policy — suppress
                logger.info(
                    f"[state-harness] Growth ratio exceeded but drift is low "
                    f"({self._drift_history[-3:]}) — suppressing at turn "
                    f"{self._turn_count}"
                )
        except BudgetExhausted:
            logger.warning(
                f"[state-harness] Budget exhausted at turn {self._turn_count}"
            )
            tripped = True

        # ── Step 6: Update telemetry ──
        self._turn_count += 1
        self._total_tokens += tokens_used

        logger.debug(
            f"[state-harness] Turn {self._turn_count}: "
            f"tokens={tokens_used}, total={self._total_tokens}, "
            f"ratio={self._guard.current_ratio or 'n/a'}, "
            f"stable={self._guard.is_stable}, "
            f"interventions={self._intervention_count}"
        )

        # If tripped (all interventions exhausted), return stop message.
        if tripped:
            assistant_message.content = (
                "I apologize, but I've reached my processing limit for this request. "
                "Please contact us again for further assistance."
            )
            assistant_message.tool_calls = None

        return assistant_message, state

    def _apply_causal_intervention(
        self, state: LLMAgentState
    ) -> LLMAgentState:
        """Apply RG compression as causal feedback (§4→§6 of theory paper).

        Instead of killing the task, aggressively compress the conversation
        history and inject a re-orientation note. This is the closed-loop
        controller: detect instability → coarse-grain → continue.

        The RG decimator retains structurally important messages (tool calls,
        policy decisions) while pruning low-relevance filler. A system note
        is prepended to help the agent re-orient after context compression.
        """
        if len(state.messages) <= 3:
            return state  # Too few messages to compress

        # 1. Extract text from all messages
        texts = []
        for msg in state.messages:
            content = getattr(msg, "content", None) or ""
            if not content:
                content = str(msg)[:200]
            texts.append(content)

        # 2. Aggressive RG compression (lower threshold = retain less)
        aggressive_rg = RGDecimator(
            threshold=0.2,  # More aggressive than normal (0.3)
            max_retained=20,  # Fewer retained messages
        )
        scored = aggressive_rg.decimate(texts)
        retained_indices = {s.index for s in scored if s.retained}

        # 3. Always keep the last 3 messages (recent context is critical)
        n = len(state.messages)
        for i in range(max(0, n - 3), n):
            retained_indices.add(i)

        original_count = len(state.messages)
        state.messages = [
            state.messages[i] for i in sorted(retained_indices)
        ]

        # 4. Inject a re-orientation note as user message
        from tau2.data_model.message import UserMessage
        reorient_msg = UserMessage(
            role="user",
            content=(
                "[System: Your conversation history has been compressed to "
                "improve efficiency. Focus on resolving the customer's core "
                "issue using the tools and context available. Avoid repeating "
                "previous steps — check tool results before re-calling.]"
            ),
        )
        # Insert after system messages but before conversation
        insert_pos = 0
        state.messages.insert(insert_pos, reorient_msg)

        self._rg_compressions += 1
        logger.info(
            f"[state-harness] Causal intervention compressed: "
            f"{original_count} → {len(state.messages)} messages"
        )

        return state

    @property
    def telemetry(self) -> dict:
        """Return a summary of state-harness telemetry for post-mortem analysis."""
        return {
            "turn_count": self._turn_count,
            "total_tokens": self._total_tokens,
            "is_stable": self._guard.is_stable,
            "is_tripped": self._guard.is_tripped,
            "is_frozen": self._guard.is_frozen,
            "baseline": self._guard.baseline,
            "current_ratio": self._guard.current_ratio,
            "adaptive_threshold": self._guard._config.ratio_threshold,
            "consecutive_escalating": self._guard.consecutive_escalating,
            "energy_history": list(self._guard.energy_history),
            "drift_history": self._drift_history,
            "rg_compressions": self._rg_compressions,
            "intervention_count": self._intervention_count,
            "max_interventions": self._max_interventions,
        }


# ── Factory function (matches τ³-bench agent factory signature) ────────


def create_harness_agent(tools, domain_policy, **kwargs):
    """Factory function for HarnessAgent v3.

    Registered with the τ³-bench registry as ``harness_agent``.
    v3 uses adaptive threshold by default (τ auto-calibrates from
    warmup variance) and causal intervention (compress-and-continue
    instead of kill-on-trip).

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
            - budget_ceiling (int): Max tokens per task.
            - ratio_threshold (float): Fallback τ (used when adaptive is off).
            - window (int): Consecutive escalating turns.
            - rg_enabled (bool): Enable RG compression.
            - budget_gate (int): Min tokens before monitor can trip.
            - adaptive (bool): Enable adaptive threshold (default True).
            - max_interventions (int): Max RG compressions before kill.
    """
    import os

    ratio_threshold = kwargs.get("ratio_threshold", DEFAULT_RATIO_THRESHOLD)
    budget_gate = kwargs.get("budget_gate", DEFAULT_BUDGET_GATE)
    rg_enabled = kwargs.get("rg_enabled", DEFAULT_RG_ENABLED)
    vsa_enabled = kwargs.get("vsa_enabled", True)

    # Environment variable overrides (highest precedence)
    # Allows threshold sweep from shell: HARNESS_RATIO_THRESHOLD=3.0 uv run tau2 ...
    env_ratio = os.environ.get("HARNESS_RATIO_THRESHOLD")
    if env_ratio is not None:
        ratio_threshold = float(env_ratio)
    env_gate = os.environ.get("HARNESS_BUDGET_GATE")
    if env_gate is not None:
        budget_gate = int(env_gate)

    # Feature toggles: HARNESS_RG=off disables RG, HARNESS_VSA=off disables VSA
    env_rg = os.environ.get("HARNESS_RG")
    if env_rg is not None:
        rg_enabled = env_rg.lower() not in ("off", "0", "false", "no")
    env_vsa = os.environ.get("HARNESS_VSA")
    if env_vsa is not None:
        vsa_enabled = env_vsa.lower() not in ("off", "0", "false", "no")

    adaptive = kwargs.get("adaptive", DEFAULT_ADAPTIVE)
    max_interventions = kwargs.get("max_interventions", 2)

    logger.info(
        f"[state-harness] Creating v3 agent: "
        f"adaptive={adaptive}, ratio_threshold={ratio_threshold}, "
        f"budget_gate={budget_gate}, max_interventions={max_interventions}, "
        f"rg={'on' if rg_enabled else 'off'}, vsa={'on' if vsa_enabled else 'off'}"
    )

    return HarnessAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        budget_ceiling=kwargs.get("budget_ceiling", DEFAULT_BUDGET_CEILING),
        ratio_threshold=ratio_threshold,
        window=kwargs.get("window", DEFAULT_WINDOW),
        budget_gate=budget_gate,
        rg_enabled=rg_enabled,
        vsa_enabled=vsa_enabled,
        adaptive=adaptive,
        max_interventions=max_interventions,
    )

