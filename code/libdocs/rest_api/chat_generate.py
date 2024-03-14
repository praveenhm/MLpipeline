import json
import os
import time

from libdocs.rest_api.client_base import OpenAIClientBase


def generate(
    chat_api_handler: OpenAIClientBase,
    output_location,
    output_file,
    iterations,
    base_user_prompt,
    examples,
    system_prompt,
):
    """
    Generate chat completions using the chat API
    args:
        chat_api_handler: OpenAIClientBase - the chat API handler
        output_location: str - the location to save the output file
        output_file: str - the name of the output file
        iterations: int - the number of completions to generate
        base_user_prompt: str - the base user prompt for the API
        examples: list - a list of examples to include in the prompt
        system_prompt: str - the system prompt for the API
    """

    full_path = os.path.join(output_location, output_file)
    os.makedirs(output_location, exist_ok=True)
    with open(full_path, "w") as file:
        for i in range(iterations):
            completion = chat_api_handler.chat_completion(
                base_user_prompt, examples, system_prompt
            )
            if completion is not None:
                file.write(json.dumps(completion) + "\n")
            if i % 100 == 99:
                time.sleep(1)
