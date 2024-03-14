from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ...types.types import LabeledChunk

# Specification of the classifier label.
ClassifierLabel = str

# ---------------------------------------------------------------------------------------------


class CrossEncoderChunkScore(BaseModel):
    """
    Class to hold the score of a labeled chunk from a cross encoder.
    """

    label_chunk: LabeledChunk = Field(..., description="The labeled chunk.")
    score: float = Field(..., description="The score for the labeled chunk.")


# ---------------------------------------------------------------------------------------------
class ClassifierPrediction(BaseModel):
    """Class to hold prediction from a classifier.

    Parameters
    ----------
    y_pred : the predicted class label, as a string.
    y_proba : predicted class probability, as a float.
    """

    y_pred: ClassifierLabel = Field(
        ..., description="The predicted class label."
    )
    y_proba: float = Field(..., description="The predicted class probability.")

    @field_validator("y_proba")
    @classmethod
    def y_proba_range(cls, v):
        if v < 0 or v > 1:
            raise ValueError("y_proba must be between 0 and 1")
        return v

    def __str__(self) -> str:
        return f"y_pred is {self.y_pred} and y_proba is {self.y_proba}"


# ---------------------------------------------------------------------------------------------
class ClassifierPredictionError(BaseModel):
    """Class to hold prediction error from a classifier.

    Parameters
    ----------
    error : the error message, as a string.
    """

    error: str = Field(
        None, description="The error message, if any, for the prediction."
    )
    top_predictions: list[ClassifierPrediction] = Field(
        ..., description="The predicted class label and probability."
    )
    ground_truth: ClassifierLabel = Field(
        ..., description="The true class label."
    )


# ---------------------------------------------------------------------------------------------
class LabeledChunkPrediction(BaseModel):
    """Class to hold prediction from a text classifier."""

    labeled_chunk: LabeledChunk = Field(..., description="The labeled chunk.")
    top_predictions: list[ClassifierPrediction] = Field(
        ..., description="The top predicted class label and probability."
    )
    top_prediction_error: Optional[ClassifierPredictionError] = Field(
        None, description="The error message, if any, for the top prediction."
    )


# ---------------------------------------------------------------------------------------------
class SubjectClassification(BaseModel):
    subject: str
    confidence: float


# ---------------------------------------------------------------------------------------------
class Classification(BaseModel):
    subject: str
    confidence: float
    topics: list[SubjectClassification]


# ---------------------------------------------------------------------------------------------
class Ledger(BaseModel):
    input: str
    pii: bool = False
    keywords: bool = False
    classifications: list[Classification] = []
