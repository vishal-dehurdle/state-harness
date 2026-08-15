"""Naive budget-cap agent for τ³-bench evaluation.

This is the "AgentBudget-style" baseline: a simple hard token cap with
no intelligence. When cumulative tokens exceed the budget, the agent
is killed immediately — regardless of whether it's making progress.

Used as a comparison point against harness_agent (state-harness v2) to
demonstrate that naive caps kill tasks indiscriminately while growth-ratio
monitoring achieves higher precision.

Usage:
    uv run tau2 run --domain airline --agent naive_cap_agent \\
        --agent-llm vertex_ai/gemini-2.5-flash \\
        --user-llm vertex_ai/gemini-2.5-flash
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMAgentStateType
from tau2.data_model.message import AssistantMessage
from tau2.environment.tool import Tool


# Default budget: same as harness_agent for fair comparison
DEFAULT_TOKEN_CAP = 100_000


class NaiveCapAgent(LLMAgent[LLMAgentState]):
    """LLMAgent with a simple hard token budget cap.

    No energy analysis, no growth-ratio detection, no drift monitoring.
    Just: if cumulative tokens > cap, raise and kill the task.

    This is what most production teams do today (e.g., AgentBudget).
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        token_cap: int = DEFAULT_TOKEN_CAP,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self._token_cap = token_cap
        self._cumulative_tokens = 0
        self._turn_count = 0

        logger.info(
            f"[naive-cap] NaiveCapAgent initialized: cap={token_cap} tokens"
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate next message with naive budget cap."""

        # Delegate to base LLMAgent
        assistant_message, state = super().generate_next_message(message, state)

        # Extract token usage
        tokens_used = 0
        if assistant_message.usage:
            tokens_used = (
                assistant_message.usage.get("completion_tokens", 0)
                + assistant_message.usage.get("prompt_tokens", 0)
            )

        self._cumulative_tokens += tokens_used
        self._turn_count += 1

        logger.debug(
            f"[naive-cap] Turn {self._turn_count}: "
            f"tokens={tokens_used}, cumulative={self._cumulative_tokens}"
        )

        # Simple check: over budget? Return a stop message instead of raising.
        # Raising an exception causes τ³-bench to record this as
        # "infrastructure_error" and retry 3x — wasting budget and
        # corrupting results. Returning a message lets the orchestrator
        # handle it as a normal agent turn → clean AGENT_STOP or USER_STOP.
        if self._cumulative_tokens > self._token_cap:
            logger.warning(
                f"[naive-cap] Budget cap exceeded at turn {self._turn_count}: "
                f"{self._cumulative_tokens} > {self._token_cap}"
            )
            # Return the last message as-is; the orchestrator will continue
            # and the user sim will eventually end the conversation.
            # The key metric (cost) is already captured.
            assistant_message.content = (
                f"I apologize, but I've reached my processing limit for this request. "
                f"Please contact us again for further assistance."
            )
            assistant_message.tool_calls = None  # Don't make more tool calls

        return assistant_message, state


def create_naive_cap_agent(tools, domain_policy, **kwargs):
    """Factory function for NaiveCapAgent."""
    return NaiveCapAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        token_cap=kwargs.get("token_cap", DEFAULT_TOKEN_CAP),
    )
