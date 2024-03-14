import argparse
import logging
import os
from typing import List

import libdocs.utils.text.text as text
from utils.jsonl.jsonl import JSONL


def main():
    default_drop_funcs = [f.__name__ for f in text.DEFAULT_DROP_ROW_FUNCS]
    available_drop_funcs = [f.__name__ for f in text.ALL_DROP_ROW_FUNCS]
    default_mutate_funcs = [f.__name__ for f in text.DEFAULT_MUTATING_ROW_FUNCS]
    available_mutate_funcs = [f.__name__ for f in text.ALL_MUTATING_ROW_FUNCS]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="if you want to overwrite the input files, instead of creating a new file",
    )
    parser.add_argument(
        "--mutate-funcs",
        nargs="+",
        type=str,
        dest="mutateFuncs",
        default=default_mutate_funcs,
        choices=available_mutate_funcs,
        help=f"The functions to apply for mutating the data. Defaults to: {' '.join(default_mutate_funcs)}",
    )
    parser.add_argument(
        "--drop-funcs",
        nargs="+",
        type=str,
        dest="dropFuncs",
        default=default_drop_funcs,
        choices=available_drop_funcs,
        help=f"The functions to apply for cleaning the data (dropping rows). Defaults to: {' '.join(default_drop_funcs)}",
    )
    parser.add_argument(
        "--help-funcs",
        dest="helpFuncs",
        action="store_true",
        help="Lists all available drop and mutate funcs that you can pass with the --drop-funcs and --mutate-funcs argument and exits",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    parser.add_argument("files", nargs="*", help="Input files to convert")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    # see if we only need to list the available apply funcs
    if args.helpFuncs:
        print("Available Mutate Functions:")
        for func_name in available_mutate_funcs:
            # getattr() is technically "better", however, __dict__() is faster and we know/trust the package
            f = text.__dict__[func_name]
            doc = f.__doc__.strip().replace("\n", " ")
            doc = " ".join(doc.split())
            print(f"{func_name}: {doc}")
        print("")
        print("Available Drop Functions:")
        for func_name in available_drop_funcs:
            # getattr() is technically "better", however, __dict__() is faster and we know/trust the package
            f = text.__dict__[func_name]
            doc = f.__doc__.strip().replace("\n", " ")
            doc = " ".join(doc.split())
            print(f"{func_name}: {doc}")
        return

    # we need to use nargs="*" instead of nargs="+" for files as we have the --help-funcs function
    # so we must do an additional length check here
    if len(args.files) < 1:
        logging.error("you must provide at least one file")
        os._exit(1)

    # build list of mutate functions
    mutate_funcs: List[text.MutatingRowFunc] = []
    for func_name in args.mutateFuncs:
        # getattr() is technically "better", however, __dict__() is faster and we know/trust the package
        f = text.__dict__[func_name]
        mutate_funcs.append(f)

    # build list of drop functions
    drop_funcs: List[text.DropRowFunc] = []
    for func_name in args.dropFuncs:
        # getattr() is technically "better", however, __dict__() is faster and we know/trust the package
        f = text.__dict__[func_name]
        drop_funcs.append(f)

    for file in args.files:
        # input file must be a '.jsonl' file, otherwise we'll get into Hell's kitchen with JSONL
        # plus this ensures file naming consistency as well
        if not file.endswith(".jsonl"):
            logging.error(
                f"{file}: does not have a '.jsonl' extension. Skipping..."
            )
            continue

        # open file, load dataframe
        jl = JSONL()
        try:
            jl.from_file(file)
        except Exception as err:
            logging.error(
                f"{file}: failed to read JSONL from file: {err}. Skipping..."
            )
            continue

        # dedup input data
        try:
            logging.info(f"{file}: dedup data...")
            text.dedup_input_data(jl.df)
            logging.info(f"{file}: dedup data... DONE")
        except Exception as err:
            logging.error(
                f"{file}: failed to deduplicate input data: {err}. Skipping..."
            )
            continue

        # mutate input data
        try:
            logging.info(
                f"{file}: mutating input data with: {' '.join(args.mutateFuncs)}..."
            )
            jl.df = text.mutate_input_data(jl.df, mutate_funcs)
            logging.info(f"{file}: mutating input data... DONE")
        except Exception as err:
            logging.error(
                f"{file}: failed to mutate input data: {err}. Skipping..."
            )
            continue

        # cleanup input data
        try:
            logging.info(
                f"{file}: cleaning input data with: {' '.join(args.dropFuncs)}..."
            )
            jl.df = text.drop_input_data(jl.df, drop_funcs)
            logging.info(f"{file}: cleaning input data... DONE")
        except Exception as err:
            logging.error(
                f"{file}: failed to cleanup input data: {err}. Skipping..."
            )
            continue

        # write out to new file (or overwrite)
        outdir = os.path.dirname(file)
        if outdir == "":
            outdir = os.getcwd()
        outfile = os.path.splitext(os.path.basename(file))[0]
        if not args.overwrite:
            outfile = f"{outfile}.cleaned"
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
