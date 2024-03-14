import io
import logging
from typing import List, Tuple

import pandas as pd
from libdocs.huggingface.huggingface import upload_csv_to_huggingface
from libdocs.types.types import LabeledChunk
from libdocs.utils.jsonl.jsonl import JSONL
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def df_to_train_df(
    df,
    text_label: str = "text",
    subject_label: str = "label",
) -> pd.DataFrame:
    """
    Convert a DataFrame to a training dataset with 5 columns.
    """
    mdf = pd.DataFrame()
    mdf["text"] = df[text_label]
    mdf["hypothesis"] = df[text_label]
    mdf["labels"] = 1
    mdf["task_name"] = ""
    mdf["label_text"] = df[subject_label]
    return mdf


def df_to_bytes(
    df: pd.DataFrame,
) -> bytes:
    """
    Convert a DataFrame to a BytesIO object (binary in-memory file) containing CSV data.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        bytes: The CSV data as bytes.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Rewind to the start of the StringIO buffer
    # Convert to bytes
    csv_bytes = csv_buffer.getvalue().encode()
    # Create a BytesIO object in binary mode
    binary_csv_buffer = io.BytesIO(csv_bytes)
    return binary_csv_buffer


def df_to_train_test_bytes(
    df: pd.DataFrame,
) -> Tuple[bytes, bytes, pd.DataFrame]:
    """
    Convert a DataFrame into train and test datasets represented as bytes.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[bytes, bytes, pd.DataFrame]: A tuple containing the train dataset as bytes,
        the test dataset as bytes, and the original test DataFrame.
    """
    # random.shuffle(df)
    df = df.sample(frac=1)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )

    logging.info(
        f" train dataset size: {len(train_df)} \n"
        f" test dataset size: {len(test_df)}"
    )
    # logging.info(df[].value_counts())
    # Convert DataFrames to bytes
    train_csv_bytes = df_to_bytes(train_df)
    test_csv_bytes = df_to_bytes(test_df)
    return train_csv_bytes, test_csv_bytes, train_df, test_df


def load_data(
    input_dir: str,
    input_filename: str,
    text_label: str = "text",
    subject_label: str = "label",
    filters={},
) -> Tuple[List[LabeledChunk], List[LabeledChunk]]:
    """
    load_data loads one or more files from a directory to provide training and test
    data in form of LabeledChunk
    """
    # Get all data
    df = JSONL().from_files(input_dir, input_filename)

    # if the 'id' does not exist, add it. we do this before we randomize etc to make sure id
    # remains the same.
    # TODO: assert if id is missing
    if "id" not in df.columns:
        df["id"] = list(range(len(df)))

    # Apply filters
    for k, v in filters:
        if k in df.columns:
            mask = df[k].isin([v])
            df = df[mask]

    # Random shuffle once before doing another in train_test_split.
    # NOTE: This is not inadvertent.
    df = df.sample(frac=1)

    # Split data for training and testing
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_chunks = []
    for index, row in train_df.iterrows():
        assert (
            text_label in row.keys()
        ), f"text label: {text_label} missing in keys: {row.keys()}"
        assert (
            subject_label in row.keys()
        ), f"subject label: {subject_label} missing in keys: {row.keys()}"
        train_chunks.append(
            LabeledChunk(
                id=row["id"], text=row[text_label], subject=row[subject_label]
            )
        )

    test_chunks = []
    for index, row in train_df.iterrows():
        assert (
            text_label in row.keys()
        ), f"text label: {text_label} missing in keys: {row.keys()}"
        assert (
            subject_label in row.keys()
        ), f"subject label: {subject_label} missing in keys: {row.keys()}"
        test_chunks.append(
            LabeledChunk(
                id=row["id"], text=row[text_label], subject=row[subject_label]
            )
        )

    return train_chunks, test_chunks, train_df, test_df


def normalize_data(
    df: pd.DataFrame,
    filter_labels: list[str],
    reduce_majority_to: float = 1.0,
    subject_label: str = "label",
) -> pd.DataFrame:
    """Normalize the distribution of the data in the DataFrame."""

    logging.info(
        f"subject_label: {subject_label}   filter_labels: {filter_labels}"
    )
    logging.info("Normalizing the distribution Count")
    logging.info(f"Total count of distribution :  {len(df)}")
    logging.info(df[subject_label].value_counts())

    # Filter the dataframe to exclude unwanted labels
    df_filtered = df[~df[subject_label].isin(filter_labels)]
    df_filtered = df_filtered.sample(frac=1)

    # Get the minimum and maximum label counts (after filtering)
    min_count = int(
        df_filtered[subject_label].value_counts().min() * reduce_majority_to
    )
    max_count = min_count * 2

    # Group data by label text
    grouped_df = df_filtered.groupby(subject_label)
    sampled_df = grouped_df.apply(
        lambda x: x.sample(min(max_count, len(x)), random_state=42)
    )

    # Normalize the data in the dataframe
    df = sampled_df.reset_index(drop=True)
    df = df.sample(frac=1)

    # Print the normalized distribution
    logging.info("After normalizing")
    logging.info(f"Total count of distribution after Normalizing:  {len(df)}")
    logging.info(df[subject_label].value_counts())

    return df


def upload_to_hf(
    train_csv_bytes,
    train_filename,
    test_csv_bytes,
    test_filename,
    input_hf_dataset,
    hf_access_token,
):
    logging.info(
        upload_csv_to_huggingface(
            train_csv_bytes,
            input_hf_dataset,
            hf_access_token,
            train_filename,
            private=True,
        )
    )
    logging.info(
        upload_csv_to_huggingface(
            test_csv_bytes,
            input_hf_dataset,
            hf_access_token,
            test_filename,
            private=True,
        )
    )
