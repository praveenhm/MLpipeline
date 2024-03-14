import glob
import json
import logging
import os

import pandas as pd
from ydata_profiling import ProfileReport


class JSONL:
    """
    JSONL helper for files and data. Use cases:
    - File Manipuation:
        - Combine multiple files into one.
        - Combine multiple files into multiple files with different number of lines.
        - Split a file into multiple.
    - Data Retreival
        - Provide contents from one or more file into a pandas DataFrame.
        - Store a pandas DataFrame into one or multiple files.
    """

    def __init__(self, df=None):
        self.df = df

    def from_file(self, path: str) -> pd.DataFrame:
        """
        Reads one JSONL file and stores it as a DataFrame within self.

        Parameters:
            path: path to the JSONL file

        Raises:
            ValueError: if path does not exist or is not a file

        Return: DataFrame
        """
        if not os.path.isfile(path):
            raise ValueError(f"path '{path}' does not exist or is not a file")
        file = os.path.abspath(path)
        return self.from_files(os.path.dirname(file), os.path.basename(file))

    def from_files(
        self, input_dir: str, filename_or_wildcard: str = "*.jsonl"
    ) -> pd.DataFrame:
        """
        Read one or more JSONL file and store as a DataFrame.

        Attributes:
            input_dir (str): Directory containing JSONL files.
            filename_or_wildcard (str): File name or wildcard to accomodate multiple files.

        Raises:
            ValueError: If `input_dir` is not a valid directory.
            IOError: If there are issues reading.

        Return: DataFrame
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory '{input_dir}' does not exist.")

        if filename_or_wildcard.find("*") >= 0:
            pattern = os.path.join(input_dir, "**", f"{filename_or_wildcard}")
        else:
            pattern = os.path.join(input_dir, f"{filename_or_wildcard}")
        files = glob.glob(pattern, recursive=True)

        # Read file(s) and generate a DataFrame
        header = []
        data = []
        for file in files:
            try:
                handle = open(file, "r")
                contents = handle.readlines()
                for i, content in enumerate(contents):
                    json_object = json.loads(content)
                    if not header and len(contents) > 0:
                        header = list(json_object.keys())
                    data.append(list(json_object.values()))
            except Exception as e:
                logging.error(f"{file}: line {i+1}: {e}")
                raise e

        self.df = pd.DataFrame(data, columns=header)
        return self.df

    def to_file(
        self,
        output_dir: str,
        output_filename: str,
        max_lines_per_file: int = -1,
    ):
        """
        Output JSONL files.

        Attributes:
            output_dir (str): Output directory to store the file.
            output_filename (str): Output filename. A suffix of '.jsonl' will be automatically added.

        Raises:
            ValueError: If `output_dir` does not exist and could not be automatically created.
            IOError: If there are issues reading.
        """

        if self.df is None:
            return

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Output a single file
        if max_lines_per_file < 0:
            output_filepath = os.path.join(
                output_dir, output_filename + ".jsonl"
            )

            try:
                self.df.to_json(output_filepath, orient="records", lines=True)
            except Exception as e:
                logging.error(f"{e}")
                raise e
            return

        # Need to split into multiple files
        index = 0
        while index < len(self.df):
            start = index
            end = min(start + max_lines_per_file, len(self.df))
            df = self.df.iloc[start:end]
            output_filepath = os.path.join(
                output_dir, f"{output_filename}-{index}.jsonl"
            )

            try:
                df.to_json(output_filepath, orient="records", lines=True)
            except Exception as e:
                logging.error(f"{e}")
                raise e

            index += max_lines_per_file

    def profile(
        self,
        filepath,
        text_label: str = "text",
        subject_label: str = "subject",
    ):
        """
        Creates a profile report for the data.

        Attributes:
            filepath (str): Output file path .
            text_label (str): Label for text in input file.
            subject_label (str): Label for subject in input file.
        """
        profile = pd.DataFrame(columns=["text", "subject"])
        profile["text"] = self.df[text_label]
        profile["subject"] = self.df[subject_label]
        profile = ProfileReport(profile)
        if not filepath.endswith(".html") and not filepath.endswith(".htm"):
            filepath = filepath + ".html"
        with open(filepath, "w") as f:
            f.write(profile.html)

    def combine(
        self,
        input_dir: str,
        input_wildcard: str,
        output_dir: str,
        output_filename: str,
    ):
        """
        Combines JSONL files in the input directory and saves them to a single file in the output directory.

        Attributes:
            input_dir (str): Directory containing JSONL files.
            input_wildcard (str): Wildcard to accomodate multiple input files.
            output_dir (str): Output directory to store the file.
            output_filename (str): Output filename. A suffix of '.jsonl' will be automatically added.

        Raises:
            ValueError: If `input_dir` or `output_dir` is not a valid directory.
            IOError: If there are issues reading or writing.
        """
        self.from_files(
            input_dir=input_dir, filename_or_wildcard=input_wildcard
        )
        self.to_file(output_dir=output_dir, output_filename=output_filename)

    def split(
        self,
        input_dir: str,
        input_filename: str,
        output_dir: str,
        output_fileprefix: str,
        max_lines_per_file: int = -1,
    ):
        """
        Splits a JSONL files in the input directory into multiple output files.

        Attributes:
            input_dir (str): Directory containing JSONL files.
            input_filename (str): Filename to split into multiple files.
            output_dir (str): Output directory to store the file.
            output_fileprefix (str): Output wildcard . A suffix of the row index followed by '.jsonl' will be automatically added.

        Raises:
            ValueError: If `input_dir` or `output_dir` is not a valid directory.
            IOError: If there are issues reading or writing.
        """
        self.from_files(
            input_dir=input_dir, filename_or_wildcard=input_filename
        )
        self.to_file(
            output_dir=output_dir,
            output_filename=output_fileprefix,
            max_lines_per_file=max_lines_per_file,
        )
