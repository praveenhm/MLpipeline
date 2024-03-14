#!/usr/bin/env python3
import argparse
import glob
import logging
import os

from libdocs.chunker.batchchunker import BatchChunker
from libdocs.google.bigquery import GoogleBigQueryProcessor
from libdocs.google.document import GooglePDFProcessor
from libdocs.google.storage import GoogleStorageProcessor
from libdocs.pipeline.stage1 import PipelineObject
from libdocs.pipeline.utils import combine_chunks, upload_combined_chunks

PDF_INPUT_FILES = ["./data/pdfs/**/*.pdf"]
SPLIT_DIR = "./data/processing/pdfs/split/"
CHUNKS_DIR = "./data/processing/pdfs/chunks/"
GCS_BASE_DIR = "./data/processing"
GCS_BUCKET = "docprocessor"
GCP_PROJECT_ID = "development-398309"
GCP_DAI_LOCATION = "us"
GCP_DAI_PROCESSOR_ID = "277f11647ef22bec"
GCP_BQ_DATASET = "training"
GCP_BQ_TABLE = "pdfs"


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
            ret += glob.glob(file + "/**/*.pdf", recursive=True)
        elif not file.endswith(".pdf"):
            logging.warning(
                f"Skipping input file: '{file}': is not a supported document (PDF)"
            )
        else:
            ret.append(file)

    return ret


def main():
    """Runs all the steps of the stage1 pipeline: from splitting PDFs to processing them through DocumentAI and storing them as JSONL"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        nargs="+",
        dest="pdf",
        default=PDF_INPUT_FILES,
        help="one or more PDF input files, supports wildcards",
    )
    parser.add_argument(
        "--split-dir",
        dest="splitDir",
        type=str,
        default=SPLIT_DIR,
        help="the directory where to store the PDFs after splitting (step 2)",
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
        "--chunks-dir",
        dest="chunksDir",
        type=str,
        default=CHUNKS_DIR,
        help="the directory where to store the chunks after processing is done from documentAI (step 5)",
    )
    parser.add_argument(
        "--gcp-project-id",
        dest="gcpProjectID",
        type=str,
        default=GCP_PROJECT_ID,
        help="the GCP project ID to use for documentAI (step 5)",
    )
    parser.add_argument(
        "--gcp-dai-location",
        dest="gcpDaiLocation",
        type=str,
        default=GCP_DAI_LOCATION,
        help="the location of the DocumentAI processor (step 5)",
    )
    parser.add_argument(
        "--gcp-dai-processor-id",
        dest="gcpDaiProcessorID",
        type=str,
        default=GCP_DAI_PROCESSOR_ID,
        help="the DocumentAI processor ID (step 5)",
    )
    parser.add_argument(
        "--gcp-bq-dataset",
        dest="gcpBqDataset",
        type=str,
        default=GCP_BQ_DATASET,
        help="the Big Query dataset to use (step 7)",
    )
    parser.add_argument(
        "--gcp-bq-table",
        dest="gcpBqTable",
        type=str,
        default=GCP_BQ_TABLE,
        help="the Big Query table to use (step 7)",
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
        help="if you want to download artifacts from the GCS bucket if they exist instead of regenerating them",
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

    # Step 1
    # NOTE: Step 1 will be implemented as a watcher. It is substituted here by the --pdf flag.
    gcs = GoogleStorageProcessor(args.gcsBucket, args.gcsBaseDir)
    gpp = GooglePDFProcessor(
        args.gcpProjectID, args.gcpDaiLocation, args.gcpDaiProcessorID
    )
    gbq = GoogleBigQueryProcessor(args.gcpBqDataset, args.gcpBqTable)
    chunker = BatchChunker
    input_objs: list[PipelineObject] = [
        PipelineObject(
            file,
            args.splitDir,
            args.chunksDir,
            gcs,
            gpp,
            gbq,
            chunker,
            args.upload,
            args.download,
        )
        for file in build_file_list(args.pdf)
    ]

    # we collect processing errors here
    errors: list[str] = []

    # iterate over all input objects over and over until there are no steps to run anymore
    has_steps = True
    step_num = 0
    while has_steps:
        # this check is needed because we remove objects from the list
        # in case of errors. So this could become an infinite loop otherwise
        if len(input_objs) == 0:
            break
        for i, obj in enumerate(input_objs):
            if i == 0:
                has_steps = not obj.is_finished()
                if not has_steps:
                    break
                step_num = obj.step_num()
                logging.info(
                    f"Step {step_num}: {obj.step_name()}. {obj.step_desc()}"
                )
            try:
                # this executes the next step
                obj.run_step()
            except Exception as err:
                logging.error(
                    f"Step {step_num} failed for: {obj.pdf_file}: unexpected {err=}, {type(err)=}"
                )
                errors.append(
                    f"{obj.pdf_dir}/{obj.pdf_file}: step {step_num}: {err}"
                )
                input_objs.remove(obj)
                continue
        logging.info(f"Step {step_num}: Success")

    # run the combine step and upload step
    # NOTE: you should have all files at the ready here, otherwise this could produces garbage output
    try:
        logging.info(
            f"Combining all JSONL files from {args.chunksDir} for DocumentAI..."
        )
        combine_chunks(args.chunksDir, "documentai")
        if args.upload:
            logging.info(
                f"Uploading combineds JSONL and Parquet files to GCS bucket '{gcs.bucket.name}' for DocumentAI..."
            )
            upload_combined_chunks(gcs, args.chunksDir, "documentai")
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
            logging.error(err)


if __name__ == "__main__":
    main()
