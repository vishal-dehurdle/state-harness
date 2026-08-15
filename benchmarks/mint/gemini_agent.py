"""Gemini LM Agent for MINT benchmark using litellm + Vertex AI."""
from .base import LMAgent
import logging
import traceback
from mint.datatypes import Action
import backoff

LOGGER = logging.getLogger("MINT")


class GeminiLMAgent(LMAgent):
    """Agent that uses Gemini via litellm for MINT benchmark evaluation."""

    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys(), "model_name is required (e.g. vertex_ai/gemini-2.5-flash)"

        # Lazy import litellm
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError("litellm is required: pip install litellm")

    @backoff.on_exception(
        backoff.fibo,
        (Exception,),
        max_tries=5,
        max_time=120,
    )
    def call_lm(self, messages):
        """Call Gemini via litellm."""
        response = self.litellm.completion(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return content, usage

    def act(self, state):
        messages = state.history
        try:
            lm_output, token_usage = self.call_lm(messages)
            for usage_type, count in token_usage.items():
                state.token_counter[usage_type] += count
            action = self.lm_output_to_action(lm_output)
            return action
        except Exception as e:
            tb = traceback.format_exc()
            LOGGER.error(f"Gemini agent error: {e}")
            return Action(f"", False, error=f"GeminiError\n{tb}")
