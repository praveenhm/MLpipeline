import json
import os

import openai


class OpenAIClientBase:
    def __init__(self, api_base_url, api_key_env_var):
        self.client = openai.OpenAI(
            base_url=api_base_url,
            api_key=os.environ[api_key_env_var],
        )

    """
    Base class for OpenAI API handlers
    args:
        api_base_url: str - the base URL for the API
        api_key_env_var: str - the environment variable containing the API key
    """

    @staticmethod
    def generate_user_prompt(base_prompt, examples):
        """
        Generate a user prompt with examples for the chat API
        args:
            base_prompt: str - the base prompt for the API
            examples: list - a list of examples to include in the prompt
        """
        example_texts = [
            '{{"subject": {0}, "text": {1}}}'.format(
                json.dumps(example["subject"]), json.dumps(example["text"])
            )
            for example in examples
        ]
        combined_examples = ", ".join(example_texts)
        combined_prompt = f"{base_prompt}\n\nExamples:\n{combined_examples}"
        return combined_prompt
