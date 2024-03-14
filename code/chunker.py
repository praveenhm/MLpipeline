#!/usr/bin/env python3
import argparse
import glob
import logging
import os
from typing import List, Tuple

from libdocs.chunker.batchchunker import BaseChunker, BatchChunker
from libdocs.google.storage import GoogleStorageProcessor
from libdocs.pipeline.stage2 import PipelineObject
from libdocs.pipeline.utils import combine_chunks, upload_combined_chunks

INPUT_FILES = ["./data/processing/pdfs/chunks/**/*.documentai.jsonl"]
CHUNKS_DIR = "./data/processing/pdfs/chunks/"
GCS_BASE_DIR = "./data/processing"
GCS_BUCKET = "docprocessor"
GCP_PROJECT_ID = "development-398309"


def build_file_list(pdfs: list[str]) -> list[str]:
    ret: list[str] = []

    # expand any globs if they are in the input arguments
    expanded_files: list[str] = []
    for pdf in pdfs:
        if pdf.find("*") >= 0:
            expanded_files += glob.glob(pdf, recursive=True)
        else:
            expanded_files.append(pdf)

    # get the real file list
    for file in expanded_files:
        if not os.path.exists(file):
            logging.warning(
                f"Skipping input file: '{file}: file or directory does not exist"
            )
        elif os.path.isdir(file):
            ret += glob.glob(file + "/**/*.documentai.jsonl", recursive=True)
        elif not file.endswith(".documentai.jsonl"):
            logging.warning(
                f"Skipping input file: '{file}': is not a supported document (DocumentAI JSONL)"
            )
        else:
            ret.append(file)

    return ret


def main():
    """Runs all the steps of the stage2 pipeline: taking DocumentAI JSONL as input, and running things through the chunker"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        dest="input",
        default=INPUT_FILES,
        help="one or more DocumentAI JSONL input files, supports wildcards",
    )
    parser.add_argument(
        "--chunker",
        dest="chunkerName",
        choices=["BatchChunker"],
        default="BatchChunker",
        help="Choose the chunker implementation to use",
    )
    parser.add_argument(
        "--gcs-bucket",
        dest="gcsBucket",
        type=str,
        default=GCS_BUCKET,
        help="the GCS bucket to upload the files",
    )
    parser.add_argument(
        "--gcs-base-dir",
        dest="gcsBaseDir",
        type=str,
        default=GCS_BASE_DIR,
        help="the local base directory for the GoogleStorageProcessor class",
    )
    parser.add_argument(
        "--chunks-dir",
        dest="chunksDir",
        type=str,
        default=CHUNKS_DIR,
        help="the directory where the chunks are stored",
    )
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="if you want to upload the combined JSONL from DocumentAI to the GCS bucket",
    )
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="if you want to download the combined JSONL from DocumentAI to the GCS bucket",
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

    # select the chunker implementation
    chunker: BaseChunker = None
    match args.chunkerName:
        case "BatchChunker":
            chunker = BatchChunker
        case _:
            logging.error(f"chunker {args.chunkerName} is not supported")
            return

    # build the stage 2 pipeline
    gcs = GoogleStorageProcessor(args.gcsBucket, args.gcsBaseDir)
    obj: PipelineObject = PipelineObject(
        build_file_list(args.input),
        args.chunksDir,
        gcs,
        chunker,
        args.upload,
        args.download,
    )

    # we collect processing errors here
    errors: List[Tuple[str, Exception]] = []

    # iterate over all input objects over and over until there are no steps to run anymore
    has_steps = True
    step_num = 0
    while has_steps:
        has_steps = not obj.is_finished()
        if not has_steps:
            break
        step_num = obj.step_num()
        logging.info(f"Step {step_num}: {obj.step_name()}. {obj.step_desc()}")
        try:
            # this executes the next step
            errs = obj.run_step()
            if errs is not None and len(errs) > 0:
                errors.extend(errs)
                logging.error(
                    f"Step {step_num} failed: encountered {len(errs)}"
                )
                continue

        except Exception as err:
            logging.error(
                f"Step {step_num} failed: unexpected {err=}, {type(err)=}"
            )
            continue
        logging.info(f"Step {step_num}: Success")

    # run the combine step and upload step
    # NOTE: you should have all files at the ready here, otherwise this could produces garbage output
    try:
        logging.info(
            f"Combining all JSONL files from {args.chunksDir} for chunker '{chunker.chunker_name()}'..."
        )
        combine_chunks(args.chunksDir, chunker.chunker_name())
        if args.upload:
            logging.info(
                f"Uploading combineds JSONL and Parquet files to GCS bucket '{gcs.bucket.name}' for chunker '{chunker.chunker_name()}'..."
            )
            upload_combined_chunks(gcs, args.chunksDir, chunker.chunker_name())
        else:
            logging.warning(
                "Skipping upload. Pass '--upload true' if you want to upload the results"
            )
    except Exception as err:
        logging.error(
            f"Combining and/or uploading of it failed: unexpected {err=}, {type(err)=}"
        )

    # all done
    logging.info("Finished")

    # print all errors
    if len(errors) > 0:
        logging.info(f"All encountered errors ({len(errors)} total):")
        for err in errors:
            logging.error(f"{err[0]}: {err[1]} ({type(err[1])})")


if __name__ == "__main__":
    main()
