from abc import ABC, abstractmethod
from typing import List


class LLMZeroShotBase(ABC):
    """
    Base class for zero-shot text classification using language models.

    Attributes:
        model_name (str): Name of the model to be used.
        cumulative_score (float): Threshold for cumulative score to decide classification cut-off.
        token (str, optional): Token for authenticated model access.
    """

    def __init__(
        self, model_name: str, cumulative_score: float = 0.8, token: str = None
    ):
        self.model_name = model_name
        self.cumulative_score = cumulative_score
        self.token = token

    @abstractmethod
    def classify(self, req) -> List[List[str]]:
        """
        Abstract method to classify input text. Subclasses should implement specific classification logic.
        """
        pass
