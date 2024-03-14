from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Specification of the classifier label.
ClassifierLabel = str

# ---------------------------------------------------------------------------------------------


class ChunkSubject(BaseModel):
    """
    Class to hold the subject of a chunk of text.
    """

    label: str = Field(..., description="The subject of the text.")
    text: str = Field(..., description="The text to be classified.")
    llm_labels: Optional[List[str]] = Field(
        None, description="The label inference from the language model."
    )


# ---------------------------------------------------------------------------------------------
class LabeledChunk(BaseModel):
    """Class to hold a chunk of text.

    As a creator of the chunk, you must provide the text, subject, and optionally the topic.
    For example, if you took a Finance textbook to create a chunk, you would provide the text
    as the chunk, the subject as Finance, and the topic as the chapter name if that is appropriate.

    Parameters
    ----------
    text : the text, as a string.
    """

    text: str = Field(..., description="The text to be classified.")
    subject: str = Field(..., description="The subject of the text.")
    topic: Optional[str] = Field(None, description="The topic of the text.")
    id: Optional[int] = Field(None, description="The id of the text chunk.")

    @field_validator("text", "subject")
    @classmethod
    def text_length(cls, v):
        if len(v) == 0:
            raise ValueError("text must be non-empty")
        return v
