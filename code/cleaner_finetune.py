import argparse
import logging
import os
import pandas as pd
from libdocs.utils.banner.banner import banner
from libdocs.utils.data_cleaner.clean_data_tfidf import clean_large_data
from libdocs.utils.jsonl.jsonl import JSONL
from libdocs.utils.training.training import (
    df_to_train_test_bytes,
    normalize_data,
    upload_to_hf,
)

logging.basicConfig(level=logging.INFO)


# create args parser
def create_args_parser():
    parser = argparse.ArgumentParser(
        description="Clean, normalize, split data, and upload to Hugging Face."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="The directory path containing the input JSONL file.",
    )
    parser.add_argument(
        "-f",
        "--input-filename",
        type=str,
        required=True,
        help="The filename of the input JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        type=str,
        required=True,
        help="Base path for the output train/test JSONL files (e.g., 'output/normalized_data.jsonl').",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=True,
        help="The name of the text column in the input JSONL.",
    )
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=True,
        help="The name of the label/subject column in the input JSONL.",
    )

    parser.add_argument(
        "-ht", "--hf-access-token", type=str, help="Hugging Face Access Token"
    )

    parser.add_argument(
        "-thf",
        "--target-hf-dataset",
        type=str,
        required=True,
        help="Target Hugging Face Dataset repository name (e.g., 'username/dataset_name').",
    )

    parser.add_argument(
        "--labels-to-filter",
        type=str,
        nargs="*",
        default=[],
        help="List of labels to exclude during normalization (optional). Example: --labels-to-filter label1 label2",
    )
    parser.add_argument(
        "--reduce-majority-to",
        type=float,
        default=1.0,
        help="Reduce majority classes during normalization to this proportion relative to minority classes (1.0 = no reduction).",
    )

    return parser


# read jsonl files, normalize and write out to new jsonl files
def read_jsonl_files(
    input_dir: str,
    input_filename: str,
    text: str,
    subject: str,
) -> pd.DataFrame:
    """Loads specific columns from JSONL files, renames them, and returns a DataFrame."""
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
    logging.info(f"Original columns found: {df.columns.tolist()}")
    cols_to_select = [col for col in columns_rename_map.keys() if col in df.columns]
    if len(cols_to_select) != len(columns_rename_map):
        missing = set(columns_rename_map.keys()) - set(cols_to_select)
        logging.warning(
            f"Input file missing expected columns: {missing}. Proceeding with available columns."
        )
        if not cols_to_select:
            raise ValueError("No required columns found in input file.")

    df = df[cols_to_select].rename(columns=columns_rename_map)
    logging.info(f"DataFrame head after renaming:\n{df.head()}")

    return df


def write_jsonl_file(df, output_file_path):
    """Writes a DataFrame to a JSONL file."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_json(output_file_path, orient="records", lines=True, force_ascii=False)
    logging.info(f"DataFrame written to JSONL file: {output_file_path}")


def cleaner_finetune(
    input_dir: str,
    input_filename: str,
    output_filename: str,
    text_col: str,
    subject_col: str,
    hf_access_token: str,
    target_hf_dataset: str,
    labels_to_filter: list[str],
    reduce_majority_to: float,
):
    """
    Runs the data cleaning, normalization, and preparation pipeline.

    Args:
        input_dir: Directory containing the input JSONL file.
        input_filename: Filename of the input JSONL.
        output_filename: Base path for saving output train/test JSONL files.
        text_col: Name of the text column in the input.
        subject_col: Name of the subject/label column in the input.
        hf_access_token: Hugging Face access token.
        target_hf_dataset: Target Hugging Face dataset repository name.
        labels_to_filter: List of labels to exclude during normalization.
        reduce_majority_to: Proportion for reducing majority classes.
    """
    args = create_args_parser().parse_args()

    banner(["Step 1: Do cosine similarity based data cleaning"])
    input_file = os.path.join(args.input_dir, args.input_filename)
    intermediate_cleaned_path = args.output_filename.replace(
        ".jsonl", "_cleaned_intermediate.jsonl"
    )
    logging.info(
        f"Running clean_large_data with input: {input_file}, output: {intermediate_cleaned_path}, subject column: {args.subject}"
    )
    df = clean_large_data(input_file, intermediate_cleaned_path, args.subject)
    logging.info(f"DataFrame head after clean_large_data:\n{df.head()}")
    if args.subject not in df.columns and "label" in df.columns:
        logging.warning(
            f"Column '{args.subject}' not found after cleaning, using 'label'. Ensure clean_large_data preserves the label column."
        )
        subject_col_for_norm = "label"
    elif args.subject in df.columns:
        subject_col_for_norm = args.subject
    else:
        raise ValueError(
            f"Label column '{args.subject}' not found in DataFrame after cleaning."
        )

    banner(["Step 2: Normalize the data in the DataFrame"])
    logging.info(
        f"Normalizing data with labels_to_filter: {args.labels_to_filter}, reduce_majority_to: {args.reduce_majority_to}, subject column: {subject_col_for_norm}"
    )
    normalized_df = normalize_data(
        df, args.labels_to_filter, args.reduce_majority_to, subject_col_for_norm
    )
    logging.info(f"DataFrame head after normalizing:\n{normalized_df.head()}")

    banner(["Step 3: Data Normalized"])
    if "text" not in normalized_df.columns or "label" not in normalized_df.columns:
        raise ValueError(
            "Normalized DataFrame must contain 'text' and 'label' columns for splitting."
        )
    train_csv_bytes, test_csv_bytes, train_df, test_df = df_to_train_test_bytes(
        normalized_df
    )

    banner(["Step 5: Upload to Hugging Face"])
    upload_to_hf(
        train_csv_bytes,
        "train.csv",
        test_csv_bytes,
        "test.csv",
        args.target_hf_dataset,
        args.hf_access_token,
    )

    banner(["Step 6: Write the split DataFrames to new JSONL files"])
    train_output_path = args.output_filename.replace(".jsonl", "_train.jsonl")
    test_output_path = args.output_filename.replace(".jsonl", "_test.jsonl")
    write_jsonl_file(train_df, train_output_path)
    write_jsonl_file(test_df, test_output_path)
    logging.info(f"Train/Test DataFrames written based on {args.output_filename}")


if __name__ == "__main__":
    args = create_args_parser().parse_args()

    cleaner_finetune(
        input_dir=args.input_dir,
        input_filename=args.input_filename,
        output_filename=args.output_filename,
        text_col=args.text,
        subject_col=args.subject,
        hf_access_token=args.hf_access_token,
        target_hf_dataset=args.target_hf_dataset,
        labels_to_filter=args.labels_to_filter,
        reduce_majority_to=args.reduce_majority_to,
    )
