# Fine-tuning Process: From Raw Data to Classification

This document explains the end-to-end process for fine-tuning a classification model using the provided codebase.

## The Two-Phase Pipeline

The fine-tuning workflow consists of two sequential phases, each implemented by a separate script:

1. **Data Preparation Phase** (`code/cleaner_finetune.py`)
2. **Model Training & Evaluation Phase** (`code/classifier_model.py`)

## Phase 1: Data Preparation

**Script**: `code/cleaner_finetune.py`  
**Purpose**: Transform raw data into a normalized, balanced dataset on Hugging Face Hub.

### Execution Example

```bash
python code/cleaner_finetune.py \
  -i data/raw_input \
  -f raw_data.jsonl \
  -o data/processed/normalized.jsonl \
  -t text_column_name \
  -s label_column_name \
  -ht YOUR_HF_TOKEN \
  -thf username/your-dataset-name \
  --reduce-majority-to 0.5
```

### Step-by-Step Process

1. **Data Loading**: Reads JSONL data from `-i` (directory) and `-f` (filename).

2. **Data Cleaning**: 
   - Applies TF-IDF similarity-based cleaning via `clean_large_data()`.
   - Preserves the label column (`-s`) during cleaning.

3. **Label Normalization**:
   - Balances class distribution with `normalize_data()`.
   - Can exclude specific labels via `--labels-to-filter`.
   - Can downsample majority classes via `--reduce-majority-to`.

4. **Train-Test Split**: 
   - Creates training and test splits with `df_to_train_test_bytes()`.
   - Maintains proportional class distribution across splits.

5. **Output & Upload**:
   - Saves processed splits locally (`_train.jsonl` and `_test.jsonl`).
   - **Critical**: Uploads dataset to Hugging Face Hub (`-thf`) as CSV files.
   - The uploaded dataset must have `train` and `test` splits with `text` and `label` columns.

## Phase 2: Model Training & Evaluation

**Script**: `code/classifier_model.py`  
**Purpose**: Fine-tune a model using the prepared dataset and evaluate its performance.

### Execution Example

```bash
python code/classifier_model.py \
  --hf-access-token YOUR_HF_TOKEN \
  --wandb-access-token YOUR_WANDB_TOKEN \
  --model-for-training-finetune "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33" \
  --input-hf-dataset "username/your-dataset-name" \
  --hf-model-name "username/your-finetuned-model" \
  --subject-column label \
  --text text \
  --output-classify-file results/evaluation_results.jsonl
```

### Step-by-Step Process

1. **Initialize & Configure**:
   - Parse command-line arguments for dataset, model names, and settings.
   - Check for CUDA GPU availability (required).
   - By default, the script performs both training and evaluation.

2. **Dataset Loading**:
   - Loads the dataset from Hugging Face Hub (`--input-hf-dataset`).
   - This dataset was created during Phase 1.
   - Extracts the test split and holds it aside for final evaluation.

3. **Fine-tuning Process** (handled by `finetune()` in `code/libdocs/finetune/finetune.py`):
   - **Label Preparation**: 
     - Derives `label2id` and `id2label` mappings from training data labels.
     - Applies mappings to create numerical label columns.
   - **Tokenization**:
     - Converts text to tokens with max length 512.
     - Uses tokenizer from the base model specified by `--model-for-training-finetune`.
   - **Model Initialization**:
     - Loads the pre-trained base model.
     - Configures classification head with the label mapping.
   - **Training**:
     - Uses Hugging Face `Trainer` with standard hyperparameters.
     - Trains for 2 epochs, with learning rate adjusted by model size.
     - Tracks metrics (accuracy, F1) with Weights & Biases.
   - **Model Publishing**:
     - Pushes the fine-tuned model to Hugging Face Hub at `--hf-model-name`.

4. **Evaluation Process** (handled by `test()` which calls `run()` from `code/libdocs/finetune/run.py`):
   - **Classifier Setup**:
     - Instantiates `DebertaZeroShot` with the fine-tuned model.
   - **Batch Processing**:
     - Processes test data in batches for efficiency.
   - **Prediction & Metrics**:
     - For each text, predicts labels and scores.
     - Calculates top-1, top-2, and top-3 accuracy.
     - Top-k accuracy checks if the correct label is among the k highest-ranked predictions.
   - **Results Output**:
     - Saves detailed results to `--output-classify-file`.
     - Reports accuracy metrics to console.

## Technical Details & Common Pitfalls

1. **Correct Label Mapping** (Critical):
   - The label mapping must be derived from training data and consistently applied.
   - The mapping ensures text labels are converted to numerical values for model training.
   - Previous bug (fixed): All examples were wrongly assigned label "1".

2. **Column Name Requirements**:
   - The dataset on Hugging Face Hub must have standardized column names.
   - For training: `text` and `label` columns are required.
   - For evaluation: Map input columns via `--text` and `--subject-column` arguments.

3. **Dataset Pipeline Dependencies**:
   - Phase 2 depends on Phase 1 completing successfully and uploading to Hub.
   - The dataset must be accessible with the provided authentication token.

4. **Memory Management**:
   - For large models or datasets, monitor GPU memory usage.
   - Use `--downsample` flag for quicker testing.
   - Memory management functions (`flush()`) are built in.

5. **Top-K Accuracy Calculation**:
   - The evaluation correctly calculates if the true label is within the top k predictions.
   - Fixed implementation checks specifically the first k elements (`[:k]`).

## Data Flow Diagram

```
Raw Data (JSONL) → Cleaned Data → Normalized Data → 
  → HF Dataset (train/test) → Fine-tuned Model → Evaluation Metrics
```

## Core Functions

- `clean_large_data()`: TF-IDF based similarity cleaning
- `normalize_data()`: Balance class distribution
- `df_to_train_test_bytes()`: Create train/test splits
- `finetune()`: Core fine-tuning logic
- `run()`: Evaluation and metrics calculation

Each of these functions can be configured with appropriate parameters via the respective script's command line arguments. 