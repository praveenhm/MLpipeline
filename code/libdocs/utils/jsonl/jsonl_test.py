import json
import os
import tempfile

import pandas as pd
import pytest
from jsonl import JSONL  # Assuming the class is named JSONL in jsonl.py


@pytest.fixture
def setup_jsonl_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sample JSONL content
        sample_data = (
            '{"name": "John Doe", "age": 30}\n{"name": "Jane Doe", "age": 25}'
        )

        # Create a sample JSONL file
        file_path = os.path.join(temp_dir, "sample.jsonl")
        with open(file_path, "w") as file:
            file.write(sample_data)

        # Yield the temporary directory path for the test case to use
        yield temp_dir


def test_read_single_jsonl_file_using_from_file(setup_jsonl_files):
    jsonl = JSONL()
    input_dir = setup_jsonl_files
    path = os.path.join(input_dir, "sample.jsonl")
    jsonl.from_file(path)

    assert (
        not jsonl.df.empty
    ), "DataFrame should not be empty after reading a JSONL file"
    assert (
        len(jsonl.df) == 2
    ), "DataFrame should contain two rows corresponding to the JSONL file content"


def test_read_single_jsonl_file_using_from_files(setup_jsonl_files):
    jsonl = JSONL()
    input_dir = setup_jsonl_files
    jsonl.from_files(input_dir=input_dir, filename_or_wildcard="sample.jsonl")

    assert (
        not jsonl.df.empty
    ), "DataFrame should not be empty after reading a JSONL file"
    assert (
        len(jsonl.df) == 2
    ), "DataFrame should contain two rows corresponding to the JSONL file content"


def test_nonexistent_directory():
    jsonl = JSONL()
    with pytest.raises(ValueError):
        jsonl.from_files(
            input_dir="nonexistent_dir", filename_or_wildcard="sample.jsonl"
        )


def test_read_jsonl_files_with_wildcard(setup_jsonl_files):
    # Assuming another JSONL file is also created in the setup_jsonl_files fixture
    jsonl = JSONL()
    input_dir = setup_jsonl_files
    jsonl.from_files(input_dir=input_dir, filename_or_wildcard="*.jsonl")

    # The exact assertion might depend on the number of files and their content
    assert (
        not jsonl.df.empty
    ), "DataFrame should not be empty after reading JSONL files with a wildcard"


def test_store_dataframe_into_jsonl(setup_jsonl_files):
    jsonl = JSONL()
    output_dir = setup_jsonl_files
    output_file = "df_output"

    # Create a sample DataFrame
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 24]})
    jsonl.df = df  # Assuming the DataFrame is stored in an attribute named df

    # Assuming a method name store_dataframe for demonstration
    jsonl.to_file(output_dir=output_dir, output_filename=output_file)

    assert os.path.exists(
        os.path.join(output_dir, output_file + ".jsonl")
    ), "Output file should exist after storing DataFrame"
    # Further assertions can check the content of the output file against the original DataFrame


def test_combine_files_into_one(setup_jsonl_files):
    jsonl = JSONL()

    dir = setup_jsonl_files
    output_filename = "combined"

    # Assuming a method name combine_files for demonstration
    jsonl.combine(
        input_dir=dir,
        input_wildcard="*.jsonl",
        output_dir=dir,
        output_filename=output_filename,
    )

    assert os.path.exists(
        os.path.join(dir, output_filename + ".jsonl")
    ), "Combined file should exist after merging"
    # Further assertions can include checking the number of lines in the output file


def test_split_file_into_multiple(setup_jsonl_files):
    jsonl = JSONL()
    input_dir = setup_jsonl_files
    input_filename = "sample.jsonl"
    output_dir = input_dir
    output_prefix = "test"
    lines_per_file = (
        100  # Specifying the desired number of lines per split file
    )

    # Creating a large sample JSONL file for splitting
    with open(os.path.join(input_dir, input_filename), "w") as f:
        for i in range(
            500
        ):  # Assume 500 lines to create a sufficiently large file
            f.write(
                json.dumps({"index": i, "value": f"Sample data {i}"}) + "\n"
            )

    # Calling the split method
    jsonl.split(
        input_dir=input_dir,
        input_filename=input_filename,
        output_dir=output_dir,
        output_fileprefix=output_prefix,
        max_lines_per_file=lines_per_file,
    )

    # Verify the creation of the expected number of output files
    expected_files_count = (500 // lines_per_file) + (
        1 if 500 % lines_per_file != 0 else 0
    )
    output_files = [
        f
        for f in os.listdir(output_dir)
        if f.endswith(".jsonl") and f != input_filename
    ]

    print(output_files)

    assert (
        len(output_files) == expected_files_count
    ), "Incorrect number of files created after splitting"

    # Further verification can include checking the line count of each output file
    for output_file in output_files:
        with open(os.path.join(output_dir, output_file), "r") as f:
            lines = f.readlines()
            assert (
                len(lines) <= lines_per_file
            ), f"File {output_file} contains more than {lines_per_file} lines"
