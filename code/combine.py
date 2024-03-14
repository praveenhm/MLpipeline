#!/usr/bin/env python3
import argparse
import logging

from libdocs.chunker.batchchunker import BatchChunker
from libdocs.google.storage import GoogleStorageProcessor
from libdocs.pipeline.utils import combine_chunks, upload_combined_chunks

CHUNKS_DIR = "./data/processing/pdfs/chunks/"
GCS_BASE_DIR = "./data/processing"
GCS_BUCKET = "docprocessor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-dir",
        dest="chunksDir",
        type=str,
        default=CHUNKS_DIR,
        help="the directory where to store the chunks after processing is done from documentAI (step 5)",
    )
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="set to true if you want to upload the results to the GCS bucket",
    )
    parser.add_argument(
        "--gcs-bucket",
        dest="gcsBucket",
        type=str,
        default=GCS_BUCKET,
        help="the GCS bucket to upload the split PDFs to (step 3)",
    )
    parser.add_argument(
        "--gcs-base-dir",
        dest="gcsBaseDir",
        type=str,
        default=GCS_BASE_DIR,
        help="the local base directory for the GoogleStorageProcessor class (step 3)",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    gcs = GoogleStorageProcessor(args.gcsBucket, args.gcsBaseDir)
    chunker = BatchChunker

    try:
        logging.info(
            f"Combining all JSONL files from {args.chunksDir} for DocumentAI..."
        )
        combine_chunks(args.chunksDir, "documentai")
        logging.info(
            f"Combining all JSONL files from {args.chunksDir} for chunker '{chunker.chunker_name()}'..."
        )
        combine_chunks(args.chunksDir, chunker.chunker_name())
        if args.upload:
            logging.info(
                f"Uploading combineds JSONL and Parquet files to GCS bucket '{gcs.bucket.name}' for DocumentAI..."
            )
            upload_combined_chunks(gcs, args.chunksDir, "documentai")
            logging.info(
                f"Uploading combineds JSONL and Parquet files to GCS bucket '{gcs.bucket.name}' for chunker '{chunker.chunker_name()}'..."
            )
            upload_combined_chunks(gcs, args.chunksDir, chunker.chunker_name())
        else:
            logging.info(
                "Skipping upload. Pass '--upload true' if you want to upload the results"
            )
    except Exception as err:
        logging.error(
            f"Combining and/or uploading of it failed: unexpected {err=}, {type(err)=}"
        )
        raise

    # all done
    logging.info("Finished")


if __name__ == "__main__":
    main()
