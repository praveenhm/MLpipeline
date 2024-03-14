import json
import logging

from libdocs.rest_api.chat_generate import generate
from libdocs.rest_api.client_base import OpenAIClientBase
from libdocs.rest_api.prompt import base_user_prompt, examples, system_prompt
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Chunk(BaseModel):
    text: str = Field(description="The main text content")
    subject: str = Field(description="Subject related to the text")


class ChatAPIHandler(OpenAIClientBase):
    def __init__(self, api_base_url, api_key_env_var, model):
        super().__init__(api_base_url, api_key_env_var)
        self.model = model

    """
    Class for handling the chat API
    args:
        api_base_url: str - the base URL for the API
        api_key_env_var: str - the environment variable containing the API key
        model: str - the model to use for the API
    """

    def chat_completion(self, base_user_prompt, examples, system_prompt):
        """
        Generate chat completions using the chat API
        args:
            filename: str - the name of the output file
            location: str - the location to save the output file
            iterations: int - the number of completions to generate
            base_user_prompt: str - the base user prompt for the API
            examples: list - a list of examples to include in the prompt
            system_prompt: str - the system prompt for the API
        """

        user_prompt = self.generate_user_prompt(base_user_prompt, examples)
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                response_format={
                    "type": "json_object",
                    "schema": Chunk.model_json_schema(),
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=1.0,
                top_p=0.7,
                frequency_penalty=2.0,
                stop=["\n\n"],
            )
            response_data = json.loads(
                chat_completion.choices[0].message.content
            )
            new_example = {
                "text": response_data["text"],
                "subject": response_data["subject"],
            }
            examples.append(new_example)
            if len(examples) > 10:
                examples.pop(0)
            return response_data

        except Exception as e:
            logging.info(f"Error: {e}")


if __name__ == "__main__":
    API_BASE_URL = "https://api.together.xyz/v1"
    API_KEY_ENV_VAR = "TOGETHER_API_KEY"
    MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    output_file = "outputx.jsonl"
    output_location = "./output"
    iterations = 3

    chat_api_handler = ChatAPIHandler(API_BASE_URL, API_KEY_ENV_VAR, MODEL)
    generate(
        chat_api_handler,
        output_location,
        output_file,
        iterations,
        base_user_prompt,
        examples,
        system_prompt,
    )
