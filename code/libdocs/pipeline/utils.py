import hashlib
import os

import pandas as pd
from google.storage import GoogleStorageProcessor
from utils.jsonl.jsonl import JSONL


def chunks_to_dataframe(
    chunks: list[str], subject: str, src: str
) -> pd.DataFrame:
    entity_ids: list[str] = []
    for chunk in chunks:
        entity_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        entity_ids.append(entity_id)
    df = pd.DataFrame()
    df["entity_id"] = entity_ids
    df["text"] = chunks
    df["label"] = subject
    df["input_src"] = src
    return df


def combine_chunks(chunks_dir: str, chunker_name: str):
    """
    Combines all chunks of the "chunker_name" chunker in chunks_dir into a single JSONL and Parquet file.
    """
    jl = JSONL()
    jl.from_files(
        chunks_dir + "/",
        f"*.{chunker_name}.jsonl",
    )
    # no compression is supposed to be the fastest for Big Query import
    jl.df.to_parquet(
        os.path.join(chunks_dir, f"{chunker_name}.combined.parquet"),
        compression=None,
    )
    jl.to_file(chunks_dir, f"{chunker_name}.combined")
    jl.profile(
        os.path.join(chunks_dir, f"{chunker_name}.combined.html"),
        "text",
        "label",
    )


def upload_combined_chunks(
    gcs: GoogleStorageProcessor, chunks_dir: str, chunker_name: str
):
    """
    Uploads the combined JSONL file and Parquet file of the "chunker_name" chunker to the GCS bucket.
    """
    gcs.upload(os.path.join(chunks_dir, f"{chunker_name}.combined.parquet"))
    gcs.upload(os.path.join(chunks_dir, f"{chunker_name}.combined.jsonl"))
    gcs.upload(os.path.join(chunks_dir, f"{chunker_name}.combined.html"))
