import logging

import pandas as pd
from libdocs.classifiers.common.types import ClassificationRequest
from libdocs.classifiers.common.zeroshotbase import LLMZeroShotBase
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run(
    zero_shot_model: LLMZeroShotBase,
    test_df: pd.DataFrame,
    output_classify_file: str,
    batch_size: int,
    text_column: str,
    subj_column: str,
):
    """
    Processes the DataFrame for classification in batches and writes results to a CSV file

    Args:
        test_df (pd.DataFrame): DataFrame containing texts to classify.
        output_classify_file (str): Path to the output CSV file.
        batch_size (int): Number of rows to process in each batch.
    """

    correct_top1 = correct_top2 = correct_top3 = processed = 0
    results = []  # Store classification results temporarily

    logging.info(test_df.head())

    def process_batch(batch_data):
        nonlocal correct_top1, correct_top2, correct_top3, processed

        for _, row in batch_data.iterrows():
            processed += 1
            text, subj = row[text_column], row[subj_column]

            out, resp = zero_shot_model.classify(
                ClassificationRequest(input=text)
            )
            # matches = resp
            subj = subj.lower().strip()

            predicted_labels = [label.strip().lower() for label in resp]

            result = {
                "index": _,
                "text": text,
                "given": subj,
                "correct_top1": (
                    int(subj == predicted_labels[0]) if predicted_labels else 0
                ),
                "correct_top2": (
                    int(subj in predicted_labels[:2])
                    if len(predicted_labels) >= 2
                    else correct_top1
                ),
                "correct_top3": (
                    int(subj in predicted_labels)
                    if predicted_labels
                    else correct_top2
                ),
                "predicted": predicted_labels,  # Keep as list
                "score": out,
            }
            results.append(result)

            correct_top1 += result["correct_top1"]
            correct_top2 += result["correct_top2"]
            correct_top3 += result["correct_top3"]

    # Process Dataframe in batches
    for i in tqdm(
        range(0, len(test_df), batch_size), total=len(test_df) // batch_size
    ):  # Add tqdm
        batch_data = test_df.iloc[i : i + batch_size]
        process_batch(batch_data)

        # Convert batch results to DataFrame and write to JSONL file
        batch_df = pd.DataFrame(results)
        batch_df.to_json(
            output_classify_file,
            orient="records",
            lines=True,
            force_ascii=False,
            mode="a",  # Append to the file for subsequent batches
        )

        results = []  # Clear results for the next batch
        logging.info(f"  accuracy top1: {correct_top1}/{processed}")
        logging.info(f"  accuracy top2: {correct_top2}/{processed}")
        logging.info(f"  accuracy top3: {correct_top3}/{processed}")

    # Log final accuracies
    logging.info(
        f"Final Top 1 Accuracy: {correct_top1}/{processed} ({(correct_top1 / processed) * 100}%)"
    )
    logging.info(
        f"Final Top 2 Accuracy: {correct_top2}/{processed} ({(correct_top2 / processed) * 100}%)"
    )
    logging.info(
        f"Final Top 3 Accuracy: {correct_top3}/{processed} ({(correct_top3 / processed) * 100}%)"
    )
