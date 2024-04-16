import logging
from typing import Dict, List, Tuple, Union

import torch
from libdocs.classifiers.common.zeroshotbase import LLMZeroShotBase
from pydantic import BaseModel
from torch.utils.data import Dataset
from transformers import pipeline


class StringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ClassificationRequest(BaseModel):
    input: Union[str, List[str]]


default_labels = [
    # Main labels
    "business_development",
    "financial",
    "human_resource",
    "legal",
    "marketing",
    "sales",
    "strategy_and_planning",
    "technical",
    "cybersecurity",
    "risk_and_compliance",
    "conversation",
    "nsfw",
    # detected back to main labels
    # "irrelevant",
    # "sex",
    # "religious",
    # "politics",
]

mapped_labels = {
    "irrelevant": "conversation",
    "sex": "nsfw",
    "religious": "nsfw",
    "politics": "nsfw",
}


class DebertaZeroShot(LLMZeroShotBase):
    """ Zero-shot text classification using DeBERTa model."""

    def __init__(
        self,
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        cumulative_score=0.8,
        token=None,
    ):
        super().__init__(model_name, cumulative_score, token)
        try:
            self.pipe = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=torch.device("cuda"),
                max_model_len=23632,
                torch_dtype=torch.bfloat16,
                token=token,
            )
        except Exception as e:
            raise e

    def classify(
        self, req: ClassificationRequest
    ) -> Tuple[Dict[str, float], List[List[str]]]:
        """ Classify input text using the DeBERTa model.  """
        try:
            if isinstance(req.input, str):
                req.input = [req.input]

            # Truncate the input text if it exceeds 512 tokens
            max_length = 512
            truncated_inputs = []
            for text in req.input:
                if len(text.split()) > max_length:
                    truncated_text = " ".join(text.split()[:max_length])
                    truncated_inputs.append(truncated_text)
                else:
                    truncated_inputs.append(text)

            outputs = []
            dataset = StringDataset(truncated_inputs)# req.input
            with torch.inference_mode():
                outputs = self.pipe(dataset, default_labels, multi_label=False, )

            responses = []
            cumulative_score = 0
            for output in outputs:
                labels = []
                merged = {}
                for index, label in enumerate(output["labels"]):
                    cumulative_score += output["scores"][index]
                    label = (
                        label.strip()
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("resources", "resource")
                    )
                    # if label in mapped_labels.keys():
                    #     label = mapped_labels[label]
                    if label in merged.keys():
                        merged[label] += output["scores"][index]
                    else:
                        merged[label] = output["scores"][index]

                    labels.append(label)
                    if cumulative_score > self.cumulative_score:
                        break
                merged = {
                    label: score
                    for label, score in sorted(
                        merged.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                }
                responses = list(merged.keys())

            return merged, responses
        except Exception as e:
            logging.error(
                f"deberta classify exception: failed for {req.input}: {e}"
            )
            raise e


if __name__ == "__main__":
    if torch.cuda.is_available():
        d = DebertaZeroShot()

        req = ClassificationRequest(
            input="Is it safe to talk sex in the work place? and discuss politics? is there any religious discrimination?"
        )
        out, resp = d.classify(req)
        print(resp)
        print(out)
        assert "nsfw" in resp[0]
