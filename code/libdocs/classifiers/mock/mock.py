from typing import List

from libdocs.classifiers.common.types import ClassificationRequest


class MockModel:
    def classify(self, req: ClassificationRequest) -> List[List[str]]:
        return [["mock", "response"]] * len(req.input)
