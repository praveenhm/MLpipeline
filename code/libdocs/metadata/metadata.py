from __future__ import annotations

import json
import logging
import os
from typing import List

from pydantic import BaseModel, Field


class FileRange(BaseModel):
    start_page: int = Field(..., alias="start-page")
    end_page: int = Field(..., alias="end-page")


class FileMetadata(BaseModel):
    read: List[FileRange] = []


class Metadata:
    def __init__(self, filepath, not_exists_ok=False):
        if not os.path.exists(filepath):
            if not_exists_ok:
                metadata = {"read": [{"start-page": 0, "end-page": 100000}]}
                json_metadata = json.dumps(metadata, indent=4)
                with open(filepath, "w") as out:
                    out.write(json_metadata)
            else:
                assert f"metadata file: {filepath} does not exist"

        file = open(filepath, "r")
        assert file

        # Read the file
        content = file.read()
        try:
            self.meta = FileMetadata.model_validate_json(content)
        except Exception as e:
            logging.error(f"metadata file: {filepath} bad content {e}")
            raise

    def __str__(self) -> str:
        return f"{len(self.meta.read)}"

    def allow_page(self, page_number: int) -> bool:
        for r in self.meta.read:
            if page_number >= r.start_page and page_number < r.end_page:
                return True
        return False


if __name__ == "__main__":
    m = Metadata("")
