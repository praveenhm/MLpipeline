from typing import List, Union

from pydantic import BaseModel


class ClassificationRequest(BaseModel):
    input: Union[str, List[str]]
