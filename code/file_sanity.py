import logging
import os

from libdocs.metadata.metadata import Metadata


def main(input_directory: str, file_extension: str) -> int:
    import glob

    files_dict = {}

    files = glob.glob(input_directory + "**/*" + file_extension, recursive=True)

    counter: int = 0
    for file in files:
        logging.info(f"Checking file name: {file}...")
        newname = file.lower()
        newname = newname.replace(" ", "_")
        newname = newname.replace("(", "")
        newname = newname.replace(")", "")
        newname = newname.replace("&", "and")
        newname = newname.replace("+", "_")
        newname = newname.replace(" ", "_")
        newname = newname.replace("+", "_")
        newname = newname.replace("-", "_")
        newname = newname.replace("__", "_")
        newname = newname.replace("__", "_")
        newname = newname.replace("._print", "")

        name = newname.replace(".pdf", "").replace(".metadata.json", "")
        files_dict[name] = newname

        if file == newname:
            continue

        logging.info(f"Renaming: {file} to {newname}")
        counter += 1
        if os.path.exists(newname):
            logging.warning(
                f"{newname} already exists, removing this file first!"
            )
            os.remove(newname)
        os.rename(file, newname)

    for k in files_dict.keys():
        if os.path.exists(k + ".pdf"):
            if not os.path.exists(k + ".metadata.json"):
                logging.error(f"missing metadata: {k}")

    for k in files_dict.keys():
        if os.path.exists(k + ".metadata.json"):
            if not os.path.exists(k + ".pdf"):
                logging.error(f"missing metadata: {k}")

    for k in files_dict.keys():
        logging.info(f"Checking metadata: {k}.metadata.json")
        m = Metadata(k + ".metadata.json")
        logging.info(f"Metadata: {m}")

    return counter


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        default="data/pdfs/",
        help="input directory to fix filenames.",
    )
    parser.add_argument(
        "-e",
        "--file-extension",
        type=str,
        default=".*",
        help="file extension to scan for.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
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

    start_time = time.time()
    counter = main(args.input_directory, args.file_extension)
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(
        f"Total processing time: {total_time:.2f} seconds for renaming {counter} files"
    )
