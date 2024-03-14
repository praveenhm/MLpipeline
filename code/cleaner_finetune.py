import argparse
import logging
import os
import pandas as pd
from libdocs.utils.banner.banner import banner
from libdocs.utils.data_cleaner.clean_data_tfidf import clean_large_data
from libdocs.utils.jsonl.jsonl import JSONL
from libdocs.utils.training.training import (df_to_train_test_bytes,
                                             normalize_data, upload_to_hf)
logging.basicConfig(level=logging.INFO)

# create args parser
def create_args_parser():
    parser = argparse.ArgumentParser(
        description="Normalize the data in a JSONL file to a desired distribution."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        # type=str,
        required=True,
        default="data/combined/run1",
        help="The dir path to the JSONL file to normalize.",
    )
    parser.add_argument(
        "-f",
        "--input-filename",
        # type=str,
        required=True,
        default="chunktext_combined_mistral_consensus_data.jsonl",
        help="The path where the normalized JSONL file will be read.",
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        # type=str,
        required=True,
        default="output/balanced.jsonl",
        help="A JSON string representing a map of columns to rename.",
    )
    parser.add_argument(
        "-t",
        "--text",
        # type=str,
        required=True,
        default="text",
        help="The name of the new column to add.",
    )
    parser.add_argument(
        "-s",
        "--subject",
        # type=str,
        required=True,
        default="label",
        help="The value to assign to the new column.",
    )

    parser.add_argument(
        "-ht", "--hf-access-token", help="Hugging Face Access Token"
    )

    parser.add_argument(
        "-hf",
        "--input-hf-dataset",
        default="penma/all_combined",
        help="Input Hugging Face Dataset",
    )

    return parser


# read jsonl files, normalize and write out to new jsonl files
def read_jsonl_files(
    input_dir: str,
    input_filename: str,
    text: str,
    subject: str,
) -> pd.DataFrame:

    df = JSONL().from_files(input_dir, input_filename)
    if text and subject:
        logging.info(f"Using text label: {text} and subject label: {subject}")
    else:
        logging.error(
            "Please provide text label and subject label for the input document"
        )
        exit()

    columns_rename_map = {
        text: "text",
        subject: "label",
    }
    logging.info(df.head())
    # Filter columns based on keys of the columns_rename_map and rename them
    df = df[list(columns_rename_map.keys())].rename(columns=columns_rename_map)

    return df


def write_jsonl_file(df, output_file_path):
    df.to_json(output_file_path, orient="records", lines=True)
    logging.info(f" JSONL file {output_file_path}")


def transform_dataframe(input_df):
    # Create a new DataFrame with the specified columns for fine-tuning
    new_df = pd.DataFrame(
        {"text": input_df["text"], "label_text": input_df["label"], "label": 1}
    )
    return new_df


def cleaner_finetune(
    labels_to_filter: list[str], reduce_majority_to: float = 1.0
):

    args = create_args_parser().parse_args()

    # Step 1: Cosine similarity
    banner(
        [
            "Step 1: Do cosine similarity based data cleaning and finetune the model"
        ]
    )
    # input_file = args.input_dir + "/" + args.input_filename
    input_file = os.path.join(args.input_dir, args.input_filename)
    output_file = args.output_filename
    subject = args.text
    df = clean_large_data(input_file, output_file, subject)

    # # Step 1: Load the JSONL file into a DataFrame and fix the columns
    # df = read_jsonl_files(
    #     args.input_dir, args.input_filename, args.text, args.subject
    # )

    # Step 2: Normalize the data in the DataFrame
    banner(["Step 2: Normalize the data in the DataFrame"])
    normalized_df = normalize_data(
        df, labels_to_filter, reduce_majority_to, args.subject
    )

    # Step 3: transform the dataframe
    banner(["Step 3: Transform the DataFrame"])
    normalized_df = transform_dataframe(normalized_df)
    logging.info(normalized_df.head())

    # Step 4: Split the data into train and test sets
    banner(["Step 4: Splitting data into train and test sets"])
    train_csv_bytes, test_csv_bytes, train_df, test_df = df_to_train_test_bytes(
        normalized_df
    )

    # Step 5: Upload CSVs bytes to Hugging Face Datasets
    banner(["Step 5: Upload to Hugging Face"])
    upload_to_hf(
        train_csv_bytes,
        "train.csv",
        test_csv_bytes,
        "test.csv",
        args.input_hf_dataset,
        args.hf_access_token,
    )

    # Step 6: Write the DataFrame to a new JSONL file
    banner(["Step 6: Write the DataFrame to a new JSONL file"])
    write_jsonl_file(
        train_df, args.output_filename.replace(".jsonl", "_train.jsonl")
    )
    write_jsonl_file(
        test_df, args.output_filename.replace(".jsonl", "_test.jsonl")
    )
    logging.info(f"DataFrame written to {args.output_filename}")


if __name__ == "__main__":

    labels_to_filter = [
        # "business_development",
        # "business_ethics",
        # "risk_and_compliance",
        # "conversation",
    ]  # temporary fix, may not required
    reduce_majority_to = 0.5
    cleaner_finetune(labels_to_filter, reduce_majority_to)
