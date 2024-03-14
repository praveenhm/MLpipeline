import argparse
import logging
import os
from typing import List

from libdocs.utils.jsonl.jsonl import JSONL

SANITY_COLUMNS: List[str] = [
    "entity_id",
    "input_src",
    "label",
    "text",
]

KNOWN_COLUMNS: List[str] = [
    "id",
    "entity_id",
    "input_src",
    "label",
    "text",
    "deberta_labels",
    "deberta_verdict",
    "mistral_labels",
    "mistral_verdict",
    "zephyr_labels",
    "zephyr_verdict",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="if you want to overwrite the input files, instead of creating a new file",
    )
    parser.add_argument(
        "--drop-unknown-columns",
        dest="dropUnknownColumns",
        action="store_true",
        help="if you want to drop unknown columns, otherwise there will be just a warning logged",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    parser.add_argument("files", nargs="+", help="Input files to convert")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    for file in args.files:
        # input file must be a '.jsonl' file, otherwise we'll get into Hell's kitchen with JSONL
        # plus this ensures file naming consistency as well
        if not file.endswith(".jsonl"):
            logging.error(
                f"{file}: does not have a '.jsonl' extension. Skipping..."
            )
            continue

        jl = JSONL()
        try:
            jl.from_file(file)
        except Exception as err:
            logging.error(
                f"{file}: failed to read JSONL from file: {err}. Skipping..."
            )
            continue

        # rename columns that we know have changed
        jl.df.rename(
            inplace=True,
            columns={
                "input_subject": "label",
                "chunk": "text",
            },
        )

        # check that at least the minimum set of columns is in the file after renaming, skip this file otherwise
        if not all([col_name in jl.df.columns for col_name in SANITY_COLUMNS]):
            logging.error(
                f"{file}: does not have the minimum set of columns: {SANITY_COLUMNS}, existing columns: {jl.df.columns}. Skipping..."
            )
            continue

        # check for unknown columns
        for col_name in jl.df.columns:
            if col_name not in KNOWN_COLUMNS:
                logging.warning(f"{file}: unknown column: {col_name}")
                if args.dropUnknownColumns:
                    logging.info(f"{file}: dropping unknown column: {col_name}")
                    jl.df.drop(columns=[col_name], inplace=True)

        # write out to new file (or overwrite)
        outdir = os.path.dirname(file)
        if outdir == "":
            outdir = os.getcwd()
        outfile = os.path.splitext(os.path.basename(file))[0]
        if not args.overwrite:
            outfile = f"{outfile}.new"
        outpath = os.path.join(outdir, outfile + ".jsonl")  # for logging
        try:
            logging.info(
                f"{file}: writing new file to '{outpath}'. (overwrite={args.overwrite})"
            )
            jl.to_file(outdir, outfile)
        except Exception as err:
            logging.error(
                f"{file}: failed to write JSONL to file '{outpath}': {err}. Skipping..."
            )
            continue


if __name__ == "__main__":
    main()
