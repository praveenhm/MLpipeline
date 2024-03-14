import argparse
import logging
import os

from libdocs.llmchecker.junkchecker_ray import JunkChecker

GCS_BASE_DIR = ""
GCS_BUCKET = "docprocessor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        dest="inputFile",
        type=str,
        default="",
        help="Name of the file to run through the cleaner",
    )
    parser.add_argument(
        "--model-name",
        dest="modelName",
        type=str,
        default="mistral",
        choices=["mistral", "zephyr"],
        help="Name of the model to use to clean the data. Defaults to 'mistral'.",
    )
    parser.add_argument(
        "--model-instances",
        dest="modelInstances",
        default=1,
        type=int,
        help="(optional) number of model instances to run in parallel for data processing. Defaults to 1. The models need to fit into the GPU.",
    )
    parser.add_argument(
        "--output-dir",
        dest="outputDir",
        type=str,
        default=os.getcwd(),
        help="Path to the output directory where all files are being created. The naming is determined based on the input file.",
    )
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="if you want to upload the files to the GCS bucket.",
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
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    if not os.path.isfile(args.inputFile):
        raise Exception(f"{args.inputFile} does not exist or is not a file")
    if not os.path.isdir(args.outputDir):
        raise Exception(
            f"{args.outputDir} does not exist or is not a direcotry"
        )

    # TODO: add proper input file handling (from multiple files), and proper output directory storing
    logging.info(
        f"Starting cleaner using {args.modelInstances} instances of the '{args.modelName}' model, processing data from '{args.inputFile}' and storing files to '{args.outputDir}'..."  # noqa: E501
    )
    c = JunkChecker()
    c.add_model_verdict(
        args.modelName,
        args.modelInstances,
        args.inputFile,
        intermediate_dir=args.outputDir,
    )
    logging.info("Done")

    # TODO: automatically upload files when upload flag is given


if __name__ == "__main__":
    main()
