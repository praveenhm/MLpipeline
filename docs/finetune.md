# Finetuning Process Overview (`code/classifier_model.py`)

This document provides a high-level overview of the model finetuning and evaluation process implemented in `code/classifier_model.py`.

## Purpose

The script serves two main purposes:

1.  **Finetuning:** To finetune a base DeBERTa zero-shot classification model (e.g., `MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33`) on a custom dataset provided via Hugging Face Datasets.
2.  **Testing/Classification:** To evaluate the performance of either the newly finetuned model or a pre-existing model on a test dataset.

## Process Flow (with `--train` flag)

The typical workflow when initiating finetuning involves the following steps:

1.  **Argument Parsing:** The script starts by parsing command-line arguments, which configure dataset names, model identifiers, API tokens, file paths, and execution flags (`--train`).
2.  **Environment Check:** It verifies the availability of a CUDA-enabled GPU, as finetuning is resource-intensive.
3.  **Data Loading:** If `--train` is specified, it loads the specified dataset from Hugging Face Hub using the `datasets` library. The 'test' split is prepared for evaluation.
4.  **Finetuning Execution:** The `finetune_model` function is called, which in turn delegates the core training logic to `libdocs.finetune.finetune.finetune`. This involves:
    *   Setting up the Hugging Face `Trainer`.
    *   Running the training loop.
    *   Optionally logging metrics to WandB.
    *   Saving the finetuned model artifacts and potentially pushing them to Hugging Face Hub under the specified name (`--hf-model-name`).
5.  **Testing/Evaluation:** After finetuning (or if `--train` is not specified but a model name is provided), the `test` function is executed:
    *   A `DebertaZeroShot` classifier is initialized with the target model name (the newly finetuned one if training just occurred).
    *   The `libdocs.finetune.run.run` function performs inference on the test data (`test_df`) using the loaded classifier.
    *   Classification results (predictions and scores) are saved to a specified output file.

## Diagram

```mermaid
graph TD
    A[Start: Run classifier_model.py --train] --> B{Parse Args};
    B --> C{Check GPU};
    C --> D[Load HF Dataset];
    D --> E[Prepare Test Data];
    E --> F[Call finetune_model];
    F --> G[Call libdocs.finetune.finetune];
    G --> H{Train & Save Model};
    H --> I[Call test function];
    I --> J[Instantiate DebertaZeroShot w/ Finetuned Model];
    J --> K[Call libdocs.finetune.run];
    K --> L{Perform Classification};
    L --> M[Save Results to File];
    M --> Z[End];

    %% Alternative path without --train (assuming --examples or similar provides test_df)
    B --> C;
    C --> I; %% Skips D, E, F, G, H if not training
```

This flow allows for a streamlined process of taking a base model, adapting it to specific data through finetuning, and immediately evaluating its performance on a relevant test set. 