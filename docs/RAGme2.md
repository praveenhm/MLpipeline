# Explanation of `code/classifier_knn.py`

This document provides a detailed explanation of the `code/classifier_knn.py` script, which serves as a primary entry point for tasks related to K-Nearest Neighbors (KNN) classification based on text embeddings.

## Overview

The script orchestrates several key processes involved in building, evaluating, and using a KNN classifier for text data, likely related to document or chunk classification based on semantic similarity. It leverages embeddings generated from transformer models and uses FAISS for efficient similarity searching.

## Core Components and Libraries

*   **`libdocs.embedder.Embedder`**: Responsible for generating text embeddings using a specified model (e.g., Sentence Transformers like `distiluse-base-multilingual-cased-v2`). It handles loading the model and converting lists of `LabeledChunk` objects or raw text into numerical vector representations.
*   **`libdocs.faissindexer.FaissIndexer`**: Wraps the FAISS library for efficient nearest neighbor search. It's used to build an index of the training data embeddings and quickly find similar items for new inputs. It supports saving and loading the index to/from disk (`index.faiss`).
*   **`libdocs.classifiers.knn.KnnEmbeddingClassifier`**: This class likely encapsulates the logic for the KNN classification process itself. It uses an `Embedder` and a `FaissIndexer` instance (loaded from a specified index file) to perform predictions. It takes text inputs, generates embeddings, searches the FAISS index for nearest neighbors, and determines the predicted class (subject) based on the neighbors found.
*   **`libdocs.types.types.LabeledChunk`**: A data structure (likely a dataclass or NamedTuple) representing a piece of text (chunk) associated with a label (subject) and a unique ID.
*   **`libdocs.utils.jsonl.JSONL`**: Utility for reading and writing data in JSON Lines format.
*   **`libdocs.utils.training.training.load_data`**: Function to load data from JSONL files, splitting it into training and testing sets of `LabeledChunk` objects and corresponding DataFrames.
*   **`numpy`**: Used for numerical operations, especially handling embeddings as arrays.
*   **`rich`**: For formatted console output (like banners).
*   **`tqdm`**: For displaying progress bars during long operations.
*   **`argparse`**: For handling command-line arguments.
*   **`umap-learn` & `matplotlib`/`pandas` (implied by visualization functions)**: Used for dimensionality reduction (UMAP) and plotting embeddings in 2D/3D space.

## Key Functionalities

1.  **Embedding Creation (`create_embeddings`)**:
    *   Takes a model name, an index file path, and a list of `LabeledChunk` objects.
    *   Initializes the `Embedder` and `FaissIndexer`.
    *   Generates embeddings for the input chunks using the `Embedder`.
    *   Adds the generated embeddings and their corresponding chunk IDs to the `FaissIndexer`.
    *   Saves the FAISS index and the raw embeddings to disk.
    *   Returns the generated embeddings.

2.  **Data Filtering/Balancing (`filter`, `sparse_filter`, `dense_filter`)**:
    *   **Purpose**: Addresses potential class imbalance in the dataset by creating filtered subsets. Often, datasets have many examples for common subjects and few for rare ones. KNN can be biased by this. Filtering aims to create a more balanced dataset where subjects have more comparable numbers of examples.
    *   **`filter` (Main Function)**:
        *   Loads an existing `KnnEmbeddingClassifier` (which includes pre-computed embeddings and chunk data).
        *   Calculates frequency counts for each subject.
        *   Determines a target maximum frequency count (e.g., `2 * min_frequency_count`).
        *   Calls either `sparse_filter` or `dense_filter` based on the `sparse` flag and specified distance thresholds (`d`).
        *   Saves the filtered (balanced) list of `LabeledChunk` objects to a new JSONL file.
        *   Generates and saves UMAP visualizations for the filtered data.
    *   **`sparse_filter`**:
        *   Aims to create a subset where examples for each subject are spread out (sparse) in the embedding space.
        *   Iterates through chunks for each subject.
        *   For each chunk, it checks its distance to already selected chunks within the same subject using a per-subject FAISS index (`faiss_balancer`).
        *   If the chunk is sufficiently far (distance > `d`) from existing selected chunks for that subject, it's added to the balanced set.
        *   Continues until `max_per_subject` chunks are selected or all chunks are processed.
    *   **`dense_filter`**:
        *   *Implementation Note*: The provided code for `dense_filter` seems incomplete or potentially incorrect. It initializes FAISS indices but the core logic for selecting *dense* samples based on distance `d` isn't fully implemented (it trains the index but doesn't appear to use `d` for selection). It likely intends to select chunks that are *close* to cluster centroids or other dense regions, but the implementation details are missing. *Needs review/completion.*

3.  **Prediction (`predict_single_text`, `test_examples`)**:
    *   **`predict_single_text`**:
        *   Loads the `KnnEmbeddingClassifier`.
        *   Takes a single text string as input.
        *   Uses the classifier's `predict` method to get the top-K predicted subjects and their probabilities (distances).
        *   Prints the results.
    *   **`test_examples`**:
        *   Loads the `KnnEmbeddingClassifier`.
        *   Reads example texts and their true subjects from a JSONL file.
        *   Uses the classifier's `predict` method to get predictions for all examples (for K=1, 2, 3).
        *   Saves the predictions (including text, true subject, predicted subject, probability) to output JSONL files (`predictions_k1.jsonl`, etc.).
        *   Calls `summary_evaluation` to calculate and print accuracy metrics.

4.  **Evaluation (`test_and_print_accuracies`, `summary_evaluation` within `KnnEmbeddingClassifier`)**:
    *   **`test_and_print_accuracies`**:
        *   Loads the `KnnEmbeddingClassifier`.
        *   Takes a list of test `LabeledChunk` objects.
        *   Calls the classifier's `summary_evaluation` method for top-K values (1, 2, 3).
        *   Prints the calculated accuracies and the time taken for evaluation.
    *   **`summary_evaluation` (in `KnnEmbeddingClassifier`, not shown but inferred)**: This method calculates the classification accuracy by comparing the predicted subjects (top-K) against the true subjects for the provided test data.

5.  **Visualization (`create_umap`)**:
    *   Takes chunks, their embeddings, and output file paths.
    *   Uses `libdocs.embedder.visualize.create_and_plot_2d_umap` to generate a 2D UMAP plot and save it as a PNG.
    *   Uses `libdocs.embedder.visualize.create_and_save_3d_csv` to save data suitable for 3D visualization (e.g., CSV for tools like `3gs` and TSV files for TensorFlow Projector).

## Command-Line Interface (`if __name__ == "__main__":`)

The script uses `argparse` to define how it can be run from the command line. Key modes of operation determined by arguments:

*   **Default Mode (Training & Testing)**:
    *   Loads data (`load_data`).
    *   Creates embeddings and the FAISS index for the *training* data (`create_embeddings`).
    *   Tests the classifier using the *testing* data (`test_and_print_accuracies`), if `--skip-test` is not set.
    *   Generates a UMAP plot for the *training* data embeddings (`create_umap`), if `--skip-umap-generation` is not set.
*   **Filtering Mode (`--filter`, `--filter-sparse`, `--filter-dense`)**:
    *   Executes the `filter` function to create balanced subsets based on the specified `sparse` or `dense` strategy and distance thresholds. Iterates through multiple distance values.
*   **Examples Mode (`--examples`)**:
    *   Runs the `test_examples` function to generate detailed prediction outputs for a given dataset (useful for error analysis).
*   **Conversation Mode (`--conversation`)**:
    *   Runs `predict_single_text` to classify a single piece of text provided via `--conversation_text`.

**Common Arguments**:

*   `-d`/`--input-dir`: Directory containing input data.
*   `-f`/`--input-file`: Input JSONL filename (can use wildcards).
*   `-i`/`--index-file`: Path to save/load the FAISS index.
*   `-m`/`--model`: Name of the embedding model to use.
*   `-s`/`--subject-label`: Column name for the subject label in the JSONL.
*   `-t`/`--text-label`: Column name for the text content in the JSONL.
*   `-od`/`--output-dir`: Directory for output files (predictions, filtered data).

## Workflow Summary

1.  **Data Preparation**: Input data is expected in JSONL format with text and subject labels.
2.  **Training (Default Mode)**:
    *   Load and split data (train/test).
    *   Generate embeddings for training data using the chosen model.
    *   Build and save a FAISS index from training embeddings.
3.  **Testing (Default Mode / Examples Mode)**:
    *   Load the saved FAISS index and associated metadata (via `KnnEmbeddingClassifier`).
    *   Generate embeddings for test data (or single text).
    *   Perform KNN search using FAISS to find nearest neighbors in the training set.
    *   Determine predicted subject(s) based on neighbors.
    *   Calculate accuracy or output predictions.
4.  **Filtering (Filter Mode)**:
    *   Load an existing index/classifier.
    *   Apply sparse or dense filtering logic to select a balanced subset of the original data based on embedding distances.
    *   Save the filtered dataset.
5.  **Visualization**: Generate UMAP plots (2D/3D) to visually inspect the embedding space, either for the full training set or filtered subsets.

This script provides a comprehensive toolkit for experimenting with KNN classification on text embeddings, including data balancing and evaluation. 