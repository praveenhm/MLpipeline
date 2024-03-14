import json
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)


def load_valid_jsonl_to_dataframe(
    jsonl_file_path, text_column_name="text", label_column_name="label_text"
):
    """
    Load a JSONL file into a pandas DataFrame, skip

    Args:
        jsonl_file_path: str - the path to the JSONL file
    """
    valid_data = []  # List to store valid JSON objects
    with open(jsonl_file_path, "r") as file:
        for line in file:
            try:
                # Attempt to parse each line as JSON
                json_obj = json.loads(line)
                # Skip lines where label_text matches 'conversation'
                if (
                    json_obj.get(label_column_name) != "conversation"
                    and "india"
                    not in json_obj.get(text_column_name, "").lower()
                ):
                    valid_data.append(json_obj)
            except json.JSONDecodeError:
                # If a line is invalid, skip it
                continue
    return pd.DataFrame(valid_data)


def filter_rename_and_add_column_dataframe(
    dataframe, columns_rename_map, new_column_name, new_column_value
):
    # Filter columns based on keys of the columns_rename_map and rename them
    filtered_and_renamed_dataframe = dataframe[
        list(columns_rename_map.keys())
    ].rename(columns=columns_rename_map)

    # Add the new column with a fixed value
    filtered_and_renamed_dataframe[new_column_name] = new_column_value

    return filtered_and_renamed_dataframe


def write_dataframe_to_jsonl(dataframe, output_file_path):
    """
    Write a pandas DataFrame to a JSONL file.

    Parameters:
    - dataframe (DataFrame): The pandas DataFrame to write.
    - output_file_path (str): The path where the JSONL file will be saved.
    """
    dataframe.to_json(output_file_path, orient="records", lines=True)


def convert_data(
    jsonl_file_path,
    output_file_path,
    text_column_name,
    label_column_name,
    new_column_value,
):

    columns_rename_map = {
        text_column_name: "text",
        label_column_name: "label_text",
    }
    new_column_name = "source_label"
    new_column_value = new_column_value

    print(
        jsonl_file_path,
        output_file_path,
        text_column_name,
        label_column_name,
        columns_rename_map,
        new_column_name,
        new_column_value,
    )
    # Load the JSONL file into a DataFrame
    df = load_valid_jsonl_to_dataframe(
        jsonl_file_path, text_column_name, label_column_name
    )

    # Filter the DataFrame, rename columns, and add a new column
    modified_df = filter_rename_and_add_column_dataframe(
        df, columns_rename_map, new_column_name, new_column_value
    )

    # Write the modified DataFrame back as a JSONL file
    write_dataframe_to_jsonl(modified_df, output_file_path)


if __name__ == "__main__":
    # Example Usage
    jsonl_file_path = "data/inference/clean_pdf_text2.jsonl"
    output_file_path = "data/inference/clean_pdf_text2-new.jsonl"

    text_column_name = "chunk"
    label_column_name = "input_subject"
    new_column_value = "pdf_text"

    convert_data(
        jsonl_file_path,
        output_file_path,
        text_column_name,
        label_column_name,
        new_column_value,
    )
