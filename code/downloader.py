#!/usr/bin/env python3
import argparse
import logging
import os
from typing import List

from file_sanity import main as file_sanity_main
from libdocs.google.drive import (FOLDER_MIME_TYPE, GoogleDriveFile,
                                  GoogleDriveProcessor)

PDFS_DIR = "./data/pdfs/"
GOOGLE_DRIVE_ROOT_FOLDER_ID = "159Ajqu1WPZtUQH9ikMQZFCQXgx9_lsEh"
GOOGLE_DRIVE_EXCLUSION_NAMES = ["JSONL_FILES"]
CREDENTIALS = "./service-account.json"


def main():
    """Runs all the steps of the stage0 pipeline: downloads our source input PDFs from Google Drive, and runs file sanity operations on them"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdfs-dir",
        dest="pdfsDir",
        type=str,
        default=PDFS_DIR,
        help="the directory where the PDFs are going to be stored",
    )
    parser.add_argument(
        "--root-folder-id",
        dest="rootFolderID",
        type=str,
        default=GOOGLE_DRIVE_ROOT_FOLDER_ID,
        help="the ID of the root folder that we are going to download (should be the ID for the 'subject_taxonomy' folder)",
    )
    parser.add_argument(
        "--exclusion-names",
        nargs="+",
        dest="exclusionNames",
        default=GOOGLE_DRIVE_EXCLUSION_NAMES,
        help="one or more PDF input files, supports wildcards",
    )
    parser.add_argument(
        "--credentials",
        dest="credentials",
        type=str,
        default=CREDENTIALS,
        help="path to a Google Service Account credentials JSON file which has access to Google Drive",
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

    # create our GoogleDriveProcessor from the CLI arguments
    gd = GoogleDriveProcessor(
        args.rootFolderID, args.exclusionNames, args.credentials
    )

    def dl(folder_id: str, download_dir: str):
        # ensure target directory exists
        os.makedirs(download_dir, exist_ok=True)
        entries: List[GoogleDriveFile] = gd.list_entries(folder_id)
        for entry in entries:
            if entry.mimeType == FOLDER_MIME_TYPE:
                dl(entry.id, os.path.join(download_dir, entry.name))
            else:
                gd.download_id(entry.id, download_dir, entry.name)

    # download everything from the root folder
    logging.info(
        f"Downloading to {args.pdfsDir} for root folder ID '{args.rootFolderID}'..."
    )
    dl(args.rootFolderID, args.pdfsDir)
    logging.info("Download finished.")

    # now run the file sanity on the PDFs dir as well
    logging.info(f"Running file sanity on {args.pdfsDir}...")
    rename_counter = file_sanity_main(args.pdfsDir, ".*")
    logging.info(
        f"Running file sanity finished. Renamed {rename_counter} files."
    )


if __name__ == "__main__":
    main()
