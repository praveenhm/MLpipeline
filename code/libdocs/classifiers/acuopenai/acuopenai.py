import logging as logger
import os
from typing import List

import instructor
from libdocs.classifiers.acuopenai.prompt import system_prompt_constant
from libdocs.classifiers.common.types import ClassificationRequest
from openai import OpenAI
from pydantic import BaseModel, Field


class Result(BaseModel):
    subjects: List[str] = Field(..., description="The subject(s) of the text.")


TEMPERATURE = 0.1


class AcuOpenAI:
    def __init__(self, model_name="gpt-4-1106-preview"):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.openai_client = OpenAI(api_key=self.api_key)
        self.client = instructor.patch(self.openai_client)
        self.model = model_name
        self.temperature = TEMPERATURE

    def classify(self, req: ClassificationRequest) -> List[List[str]]:
        if isinstance(req.input, str):
            # Convert a single prompt to a list.
            req.input = [req.input]

        responses = []
        for input in req.input:
            logger.info(
                f"Running the chunk through the language model: {input}"
            )

            user_prompt = input
            result = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4000,
                temperature=self.temperature,
                response_model=Result,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": system_prompt_constant},
                ],
            )

            responses.append(
                [
                    label.replace(" ", "_").replace("resources", "resource")
                    for label in result.subjects
                ]
            )

        return responses
