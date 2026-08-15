"""Gemini Feedback Agent for MINT benchmark using litellm + Vertex AI."""
import re
import logging
import traceback

LOGGER = logging.getLogger("MINT")

from .gemini_agent import GeminiLMAgent
from mint.datatypes import State, Action
from mint.prompt import FeedbackPromptTemplate


class GeminiFeedbackAgent(GeminiLMAgent):
    """Feedback agent using Gemini to provide expert feedback in MINT evaluation."""

    def __init__(self, config):
        super().__init__(config)
        # The agent should not generate Assistant msg since it should provide feedback
        self.stop_words = ["\nObservation:", "\nTask:", "\nAssistant:"]
        self.feedback_prompt = FeedbackPromptTemplate()

    def lm_output_to_action(self, lm_output, form=None) -> Action:
        if form == "textual":
            feedback = lm_output
        elif form == "binary":
            first_sent = re.findall(r"([^.]*\.)", lm_output)[0]
            if "GOOD" in first_sent:
                feedback = "This is GOOD."
            elif "BAD" in first_sent:
                feedback = "This is BAD."
            else:
                raise ValueError(f"Cannot find GOOD or BAD in feedback: {lm_output}")
        else:
            feedback = lm_output
        return Action(feedback, use_tool=False)

    def act(
        self,
        state: State,
        observation: str,
        form: str,
        gt,
        task_in_context_example: str,
        tool_desc: str,
    ) -> Action:
        try:
            gt_solution = (
                (
                    f"Correct solution (please DO NOT disclose the correct solution to the assistant): {str(gt).strip()}\n"
                )
                if gt
                else "Correct solution (please DO NOT disclose the correct solution to the assistant): NOT GIVEN\n"
            )
            trajectory = (
                "---\n".join(state.history[0]["content"].split("---\n")[2:]) + "\n"
            )
            trajectory += "\n".join([x["content"] for x in state.history[1:]])
            trajectory += "\n" + observation
            trajectory = trajectory[
                trajectory.find("Task:") :
            ]
            messages = [
                {
                    "role": "user",
                    "content": self.feedback_prompt(
                        in_context_example=task_in_context_example[
                            task_in_context_example.find("Task:") :
                        ],
                        trajectory=trajectory,
                        correct_solution=gt_solution,
                        tool_desc=tool_desc,
                    ),
                }
            ]

            LOGGER.debug(
                "Feedback Agent Prompt:\n"
                + "\033[93m"
                + messages[0]["content"]
                + "\033[0m"
            )
            lm_output, token_usage = self.call_lm(messages)
            for usage_type, count in token_usage.items():
                state.token_counter["feedback_" + usage_type] += count
            action = self.lm_output_to_action(lm_output, form)
            LOGGER.debug(
                "Feedback Agent Action:\n" + "\033[91m" + action.value + "\033[0m"
            )
            return action
        except Exception as e:
            tb = traceback.format_exc()
            LOGGER.error(f"Gemini feedback error: {e}")
            return Action(f"", False, error=f"GeminiFeedbackError\n{tb}")
