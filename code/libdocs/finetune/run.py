import logging
import traceback  # Import traceback for detailed error logging

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

    # Add a warning about the output file append mode
    logging.warning(
        f"Output will be appended to {output_classify_file}. Ensure the file is empty or removed if starting a fresh run."
    )

    correct_top1 = correct_top2 = correct_top3 = processed = 0
    results = []  # Store classification results temporarily

    logging.info(test_df.head())

    def process_batch(batch_data):
        nonlocal correct_top1, correct_top2, correct_top3, processed

        for _, row in batch_data.iterrows():
            processed += 1
            text, subj = row[text_column], row[subj_column]
            subj = subj.lower().strip()  # Process subj early

            try:  # Add try block for robustness
                out, resp = zero_shot_model.classify(ClassificationRequest(input=text))

                predicted_labels = (
                    [label.strip().lower() for label in resp] if resp else []
                )

                # Calculate correctness
                is_correct_top1 = (
                    int(subj == predicted_labels[0]) if predicted_labels else 0
                )
                is_correct_top2 = (
                    int(subj in predicted_labels[:2])
                    if len(predicted_labels) >= 1
                    else 0
                )  # simplified check, handles len 1
                is_correct_top3 = (
                    int(subj in predicted_labels[:3])
                    if len(predicted_labels) >= 1
                    else 0
                )  # Corrected top-3 check

                result = {
                    "index": _,
                    "text": text,
                    "given": subj,
                    "correct_top1": is_correct_top1,
                    "correct_top2": is_correct_top2,  # Use is_correct_top2
                    "correct_top3": is_correct_top3,  # Use corrected is_correct_top3
                    "predicted": predicted_labels,  # Keep as list
                    "score": out,  # Use the score from classify
                }
                results.append(result)

                correct_top1 += result["correct_top1"]
                correct_top2 += result["correct_top2"]
                correct_top3 += result["correct_top3"]

            except Exception as e:  # Add except block
                logging.error(f"Error processing row index {_}: {e}")
                logging.error(f"Text: {text[:500]}...")  # Log problematic text snippet
                logging.error(traceback.format_exc())  # Log full traceback
                # Optionally add a placeholder result for failed rows
                results.append(
                    {
                        "index": _,
                        "text": text,
                        "given": subj,
                        "correct_top1": 0,
                        "correct_top2": 0,
                        "correct_top3": 0,
                        "predicted": ["ERROR"],
                        "score": None,  # Indicate error
                        "error": str(e),
                    }
                )

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
