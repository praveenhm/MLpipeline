# Fine-tuning Process Explanation (classifier_model.py)

This document outlines the step-by-step process for fine-tuning a classification model, primarily orchestrated by the `code/classifier_model.py` script, leveraging helper modules for data preparation, training, and evaluation.

## Overview

The overall process involves two main phases:

1.  **Phase 1: Data Preparation (using `code/cleaner_finetune.py`)**: This script takes raw input data, cleans it, normalizes the label distribution, splits it into training and testing sets, and uploads the result as a dataset to the Hugging Face Hub.
2.  **Phase 2: Model Fine-tuning and Evaluation (using `code/classifier_model.py`)**: This script takes the prepared Hugging Face dataset, fine-tunes a specified base model on the training split, saves the fine-tuned model to the Hugging Face Hub, and finally evaluates its performance on the test split.

## Phase 1: Data Preparation (`code/cleaner_finetune.py`)

This script must be run *before* the main fine-tuning script. Its purpose is to create the standardized dataset required for training.

1.  **Input**: Takes a local directory (`--input-dir`) and filename (`--input-filename`) for the raw JSONL data. Requires specifying the text (`--text`) and label (`--subject`) columns within the raw data.
2.  **Cleaning**: Applies TF-IDF based cleaning using `libdocs.utils.data_cleaner.clean_data_tfidf.clean_large_data`.
3.  **Normalization**: Balances the label distribution using `libdocs.utils.training.training.normalize_data`, potentially excluding specified labels (`--labels-to-filter`) and downsampling majority classes (`--reduce-majority-to`). This step expects the label column to be named `label` after cleaning (or uses the original `--subject` column if found).
4.  **Splitting**: Splits the normalized data into training and test sets using `libdocs.utils.training.training.df_to_train_test_bytes`. This function expects columns named `text` and `label`.
5.  **Output**: Saves the split data locally (`--output-filename`_train.jsonl / _test.jsonl) and, crucially, uploads the train and test splits as CSV files (`train.csv`, `test.csv`) to a specified Hugging Face dataset repository (`--target-hf-dataset`).

**Outcome**: A dataset suitable for fine-tuning is available on the Hugging Face Hub.

## Phase 2: Model Fine-tuning and Evaluation (`code/classifier_model.py`)

This is the main script that performs the training and evaluation.

1.  **Step 1: Setup & Argument Parsing**
    *   The script is launched from the command line.
    *   It parses arguments to configure the process, including:
        *   `--hf-access-token`, `--wandb-access-token`: API tokens.
        *   `--model-for-training-finetune`: The base model to fine-tune (e.g., a DeBERTa model from the Hub).
        *   `--input-hf-dataset`: The **name** of the Hugging Face dataset prepared in Phase 1.
        *   `--hf-model-name`: The **name** for the repository where the fine-tuned model will be saved on the Hub.
        *   `--subject-column`, `--text`: Names of the relevant columns in the *evaluation* data.
        *   `--train`: Flag to enable the fine-tuning step (default behavior if `--examples` is not set).
        *   `--examples`: Flag to run evaluation using a local file (`--filename`) instead of training first.
        *   `--eval-batch-size`: Batch size for the final evaluation.
        *   `--downsample`: Optionally use smaller subsets during fine-tuning for faster testing.
        *   `--wandb-output`: Directory for saving WandB HTML reports.
    *   Checks for CUDA availability.

2.  **Step 2: Load Evaluation Data (Conditional)**
    *   **If `--examples` is True**: Loads evaluation data (`test_df`) from a local JSONL file specified by `--input-dir` and `--filename`.
    *   **If `--examples` is False (Training Workflow)**:
        *   The script proceeds to the training step.
        *   Crucially, *before* training starts (within the `if args.train:` block), it loads the dataset specified by `--input-hf-dataset` using `datasets.load_dataset`.
        *   It extracts **only the `test` split** of this dataset into the `test_df` variable. This `test_df` is held aside for the final evaluation *after* the model has been fine-tuned.

3.  **Step 3: Fine-tuning (if `args.train` is True)**
    *   Calls the `finetune_model()` helper function.
    *   `finetune_model()` calls the core `finetune()` function from `code/libdocs/finetune/finetune.py`.
    *   Inside `finetune()`:
        *   Loads the **full dataset** (train and test splits) from the specified `--input-hf-dataset` on the Hub.
        *   Performs preprocessing:
            *   Derives `label2id` and `id2label` mappings from the *training split*.
            *   Applies the mapping to create the numerical `label` column for both train and test DataFrames.
            *   Handles optional downsampling.
            *   Tokenizes the `text` column using the tokenizer associated with `--model-for-training-finetune` (max length 512).
        *   Configures `transformers.TrainingArguments` with hyperparameters (learning rate, epochs=2, batch sizes, weight decay, etc.) and Hub integration details (`push_to_hub=True`, `hub_model_id=args.hf_model_name`).
        *   Initializes `transformers.Trainer` with the base model (`--model-for-training-finetune`), training arguments, preprocessed train/eval datasets, and a `compute_metrics` function.
        *   Executes `trainer.train()`. This trains the model on the `train` split and uses the `eval` split (derived from the original dataset's `test` split) for intermediate evaluation based on `evaluation_strategy='epoch'`.
        *   Executes `trainer.evaluate()` to get final metrics on the eval dataset.
        *   Pushes the fine-tuned model weights and tokenizer configuration to the specified Hugging Face model repository (`--hf-model-name`).
        *   Logs metrics and creates reports using WandB.

4.  **Step 4: Evaluation**
    *   This step runs after the training (if performed) or immediately if `--examples` was used.
    *   Calls the `test()` helper function.
    *   Inside `test()`:
        *   Instantiates a classifier (`DebertaZeroShot`) using the **fine-tuned model name** (`args.hf_model_name`) loaded from the Hub.
        *   Calls the `run()` function from `code/libdocs/finetune/run.py`.
    *   Inside `run()`:
        *   Takes the instantiated classifier and the `test_df` (loaded in Step 2) as input.
        *   Iterates through `test_df` using the specified `--eval-batch-size`.
        *   For each batch, it calls `zero_shot_model.classify()` on the text data.
        *   Compares the predicted labels against the ground truth (`subj_column`, which corresponds to `--subject-column`) to calculate top-1, top-2, and top-3 accuracy.
        *   Logs accuracy progress periodically.
        *   Appends detailed classification results (input text, ground truth, predictions, scores, correctness flags) to the `--output-classify-file` in JSONL format.
        *   Logs final top-k accuracies.

## Summary / Key Observations

*   **Two-Phase Process**: Data preparation (`cleaner_finetune.py`) is a distinct prerequisite step before running the main training/evaluation script (`classifier_model.py`).
*   **Hugging Face Integration**: The workflow relies heavily on the Hugging Face Hub for storing both the prepared datasets and the resulting fine-tuned models.
*   **Data Flow**: Raw data -> Cleaned/Normalized DF -> HF Dataset (Train/Test Splits) -> `finetune()` uses both splits -> `run()` uses the Test Split and the fine-tuned model.
*   **Column Naming**: Assumes specific column names (`text`, `label`) after the preparation phase for training and splitting, and relies on `--text` and `--subject-column` arguments for the final evaluation.
*   **Evaluation**: The `Trainer` performs intermediate evaluation during fine-tuning, while the separate `run` function performs the final detailed evaluation (including top-k accuracy) on the held-out test set using the *final* pushed model.
*   **Monitoring**: WandB is integrated into the `finetune` process for tracking metrics and experiment details.
*   **Modularity**: The core logic for fine-tuning (`finetune.py`) and evaluation (`run.py`) is encapsulated in separate library functions, called by the orchestration script (`classifier_model.py`). 