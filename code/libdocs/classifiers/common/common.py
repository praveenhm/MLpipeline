from typing import List

from libdocs.classifiers.common.prompt import \
    subject_taxonomy_classifier_prompt
from libdocs.classifiers.common.types import ClassificationRequest
from vllm import LLM, SamplingParams


class LLMCommon:
    def __init__(self, model_name, sampling_params=None):
        self.sampling_params = sampling_params
        if sampling_params is None:
            self.sampling_params = SamplingParams(
                temperature=0.0,
                repetition_penalty=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                logprobs=0,
                top_p=1,
                top_k=-1,
                stop="\n",
                max_tokens=1024,
            )

        self.pipe = LLM(
            model=model_name,
            max_model_len=23632,
        )

    def classify(
        self,
        req: ClassificationRequest,
        prompt=subject_taxonomy_classifier_prompt,
    ) -> List[List[str]]:
        if isinstance(req.input, str):
            # Convert a single prompt to a list.
            req.input = [req.input]
        prompts = []
        for input in req.input:
            prompts.append(
                prompt.format(
                    input=input,
                )
            )
        outputs = self.pipe.generate(
            prompts,
            self.sampling_params,
        )
        responses = []
        for output in outputs:
            label = output.outputs[0].text.strip()
            label = (
                label.replace(" ", "_")
                .replace("-", "_")
                .replace("resources", "resource")
            )
            responses.append([label])
        return responses
