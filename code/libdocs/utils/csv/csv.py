import csv
import logging
import os
from typing import List


class CSV:
    """
    Combines or splits CSV files.

    Attributes:
        input_dir (str): Directory containing CSV files.
        suffix (str): File suffix to match (`*.csv` by default).
        output_dir (str): Directory to save combined or split files.
        logger (logging.Logger): A logger instance for logging messages.

    Raises:
        ValueError: If `input_dir` or `output_dir` is not a valid directory.
        IOError: If there are issues accessing or creating files.
    """

    def __init__(self, input_dir: str, suffix: str, output_dir: str):
        """
        Initializes the `CSV` class.

        Args:
            input_dir (str): Directory containing CSV files.
            suffix (str, optional): File suffix to match (default: ".csv").
            output_dir (str, optional): Directory to save combined or split files (default: same as input_dir).
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory '{input_dir}' does not exist.")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.input_dir = input_dir
        self.suffix = suffix
        self.output_dir = output_dir

    def combine(self, file: str):
        pass

    def split(self, fileprefix: str):
        pass

    def write_to_csv(
        self, filename: str, header: List[str], data: List[List[str]]
    ):
        try:
            output_filepath = os.path.join(self.output_dir, filename)
            with open(output_filepath, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                csv_writer.writerows(data)
        except IOError as e:
            logging.error(
                f"Error opening or writing to CSV file {output_filepath}: {e}"
            )
            raise e
