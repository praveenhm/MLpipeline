import argparse
import logging
import os

import libdocs.utils.label.label as label_util
import pandas as pd
from libdocs.google.storage import GoogleStorageProcessor
from libdocs.utils.jsonl.jsonl import JSONL

GCS_BASE_DIR = ""
GCS_BUCKET = "docprocessor"

CORRECT_VERDICT = "correct"
PARTIALLY_CORRECT_VERDICT = "partially correct"
INCORRECT_VERDICT = "incorrect"


def verdict(
    df: pd.DataFrame,
    input_label_col: str,
    model_label_col: str,
    verdict_label_col: str,
):
    verdict_entries = []
    for _, row in df.iterrows():
        if row[input_label_col] == row[model_label_col][0]:
            verdict_entries.append(CORRECT_VERDICT)
        elif row[input_label_col] in row[model_label_col]:
            verdict_entries.append(PARTIALLY_CORRECT_VERDICT)
        else:
            verdict_entries.append(INCORRECT_VERDICT)
    df[verdict_label_col] = verdict_entries


def sanitize_data(df: pd.DataFrame, input_label_col: str, model_label_col: str):
    subjects = df[input_label_col].unique().tolist()
    subjects.append("conversation")
    subjects.append("irrelevant")
    subjects.append("not_safe_for_workplace")

    # check if labels can be fixed
    modified_labels = []
    for labels in df[model_label_col]:
        row_labels, discovered_labels = label_util.sanitize(subjects, labels)
        modified_labels.append(row_labels)

    df[model_label_col] = modified_labels


def need_sorting(src: pd.DataFrame, target: pd.DataFrame) -> bool:
    """
    Returns true if the rows of the target data frame are out of order and need to be sorted
    """
    return all(src["entity_id"].values == target["entity_id"].values)


def sort_rows(src: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the rows of the target dataframe according to the src data frame.
    The entity_id is taken as the reference. It returns the sorted data frame.
    """
    ret = target.set_index("entity_id")
    ret = ret.reindex(index=src["entity_id"])
    ret = ret.reset_index()
    return ret


def main():
    """
    Reads all combined model files, and adds verdict columns.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-datasets",
        nargs="+",
        dest="datasets",
        type=str,
        default=["pdfs"],
        help="All the datasets to consider while creating the combined model verdict file",
    )
    parser.add_argument(
        "--input-chunker-name",
        dest="chunkerName",
        type=str,
        default="batchchunker",
        help="Name of the chunker which determines all the input files to take into account while combining. If no chunker was used, the chunker part of the file name must be 'none'.",  # noqa: E501
    )
    parser.add_argument(
        "--input-cleaner-names",
        nargs="+",
        dest="cleanerNames",
        default=["deberta", "mistral", "zephyr"],
        help="The names of all the cleaner input files to take into account while combining.",
    )
    parser.add_argument(
        "--output-dest-file",
        dest="destFile",
        type=str,
        default="model-verdict",
        help="Path to the destination where the combined data will be stored. The '.jsonl' ending will be added automatically.",
    )
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="if you want to upload the combined JSONL to the GCS bucket",
    )
    parser.add_argument(
        "--gcs-bucket",
        dest="gcsBucket",
        type=str,
        default=GCS_BUCKET,
        help="the GCS bucket to upload the files",
    )
    parser.add_argument(
        "--gcs-base-dir",
        dest="gcsBaseDir",
        type=str,
        default=GCS_BASE_DIR,
        help="the local base directory for the GoogleStorageProcessor class",
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

    if len(args.cleanerNames) == 0:
        logging.error("at least one cleaner name needs to be provided")
        return 1

    all_df = None
    for prefix in args.datasets:
        combined_df = None
        for model_name in args.cleanerNames:
            logging.info(
                f"Dataset {prefix}: calculating verdict and combining for cleaner: {model_name}"
            )
            file = os.path.join(
                os.getcwd(),
                f"{prefix}.{args.chunkerName}.combined.{model_name}.jsonl",
            )
            # if a chunker was not in use, we are also going to try if there are 'none' files
            if not os.path.exists(file):
                file = os.path.join(
                    os.getcwd(),
                    f"{prefix}.none.input.{model_name}.jsonl",
                )
            dir = os.path.dirname(file)
            jl = JSONL()
            jl.from_files(dir, file)
            df = jl.df

            # ensure we have the model's labels columns
            col_name = f"{model_name}_labels"
            if col_name not in df.columns:
                raise RuntimeError(
                    f"{model_name}: {col_name} not found in model JSONL file: {file}"
                )

            # ensure verdict column doesn't exist in this file yet
            col_verdict_name = f"{model_name}_verdict"
            if col_verdict_name in df.columns:
                raise RuntimeError(
                    f"{model_name}: {col_verdict_name} already exists in model JSON file: {file}"
                )

            # sanitize labels first
            sanitize_data(df, "label", col_name)

            # then compute verdict
            verdict(df, "label", col_name, col_verdict_name)

            if combined_df is None:
                # we just simply go to the 2nd entry
                combined_df = df
                # keep it sorted
                combined_df = combined_df.sort_values("entity_id")
                continue

            # sort the new df
            df = df.sort_values("entity_id")

            assert len(df) == len(
                combined_df
            ), f"df: {len(df)} combined_df: {len(combined_df)}"
            assert (
                df.iloc[0].text == combined_df.iloc[0].text
            ), f"df.text: {df.iloc[0].text} combined_df.text: {df.iloc[0].text}"
            assert (
                df.iloc[0].entity_id == combined_df.iloc[0].entity_id
            ), f"df.entity_id: {df.iloc[0].entity_id} combined_df.entity_id: {df.iloc[0].entity_id}"

            # now combine them after we can trust that the column values will be in the right order
            combined_df[col_name] = df[col_name]
            combined_df[col_verdict_name] = df[col_verdict_name]

        # concatenate the datasets together
        if all_df is None:
            all_df = combined_df
        else:
            all_df = pd.concat([all_df, combined_df], ignore_index=True)

        # add 'id' column which is simply all rows enumerated (delete it if it exists already, so that we can redo it)
        # NOTE: this does not need to be a stable ID, so we are okay to regenerate this even when the files are combined out of order
        if "id" in combined_df.columns:
            combined_df.drop(inplace=True, columns=["id"])
        combined_df["id"] = [i + 1 for i in range(len(df))]

        # write out combined verdict file first
        jl = JSONL(combined_df)
        out_dir = os.getcwd()
        if args.destFile.startswith("/"):
            out_dir = os.path.dirname(args.destFile)
        out_file = f"{prefix}.{os.path.basename(args.destFile)}"
        logging.info(
            f"Writing '{prefix}' combined file: {os.path.join(out_dir, out_file)}"
        )
        jl.to_file(out_dir, out_file)

    # potentially redo the `id` column which is simply all rows enumerated
    # NOTE: this does not need to be a stable ID, so we are okay to regenerate this even when the files are combined out of order
    if "id" in all_df.columns:
        all_df.drop(inplace=True, columns=["id"])
    all_df["id"] = [i + 1 for i in range(len(all_df))]

    # write out all verdict file
    jl = JSONL(all_df)
    out_dir = os.getcwd()
    if args.destFile.startswith("/"):
        out_dir = os.path.dirname(args.destFile)
    out_file = os.path.basename(args.destFile)
    logging.info(
        f"Writing all concatenated file: {os.path.join(out_dir, out_file)}"
    )
    jl.to_file(out_dir, out_file)

    # TODO: upload the single combined files as well
    if args.upload:
        upload_file = os.path.join(out_dir, out_file + ".jsonl")
        if not os.path.isfile(upload_file):
            raise Exception(
                f"Upload file '{upload_file}' does not exist or is not a file"
            )
        logging.info("Instantiating GCS processor...")
        gcs = GoogleStorageProcessor(args.gcsBucket, args.gcsBaseDir)
        gcs.upload(upload_file)
    logging.info("Done")


if __name__ == "__main__":
    main()
