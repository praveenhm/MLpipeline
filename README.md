
# PDF Processing Pipeline with RAG and Fine-tuning

**Deepwiki documentation**<https://deepwiki.com/praveenhm/MLpipeline/1-mlpipeline-overview>

## Overview 📜

This repository implements a multi-stage machine learning pipeline designed primarily for processing PDF documents. The core functionality revolves around extracting text, chunking it intelligently, generating embeddings, indexing them for efficient retrieval, and utilizing various language models (including fine-tuned ones) for tasks like classification or analysis. It leverages cloud infrastructure (GCP, including GCS, DocumentAI, and BigQuery) extensively.

## Pipeline Stages & Data Flow ⚙️

The pipeline executes in sequential stages:

1.  **Stage 0 (Input Retrieval):** Downloads PDFs and associated metadata (`.metadata.json`) from Google Drive, performing basic file sanity checks. Source: `code/downloader.py`.
2.  **Stage 1 (Initial Ingestion & Text Extraction):**
    *   Splits PDFs into a format suitable for Google DocumentAI (`code/splitter`).
    *   Uploads split files to GCS.
    *   Utilizes Google DocumentAI for OCR and text extraction, producing `.documentai.jsonl` files containing structured text chunks.
    *   Uploads DocumentAI output back to GCS. Source: `code/ingestion.py`.
3.  **Stage 2 (Chunking):**
    *   Processes the DocumentAI output through a custom chunking mechanism (`code/chunker`, implemented in `code/libdocs/chunker/`), creating potentially more semantically meaningful or task-specific chunks (`.batchchunker.jsonl`).
    *   Uploads these refined chunks to GCS. Source: `code/chunker.py`.
4.  **Stage 3 (Storage, Analysis & Verdicts):**
    *   Stores the processed chunk data in Google BigQuery for structured access and analysis.
    *   This stage integrates the core RAG and fine-tuning capabilities.

## Retrieval-Augmented Generation (RAG) 🔍➡️🧠

The pipeline implements the core components necessary for RAG, enabling models to answer queries based on the ingested document content:

1.  **Retrieval Components:**
    *   **Chunking:** Stages 1 and 2 prepare text chunks. The custom chunker (`code/libdocs/chunker/`) is vital for creating retrieval-optimized segments.
    *   **Embedding:** Chunks are converted into vector embeddings using models from `code/libdocs/embedder/` (potentially leveraging `code/libdocs/huggingface/` or cloud APIs). These vectors capture semantic meaning.
    *   **Indexing:** A FAISS index (`code/libdocs/faissindexer/`) is built from embeddings, allowing highly efficient vector similarity searches across the document corpus.
    *   **Search:** Vector search capabilities are used to find the most relevant chunks (top-k) based on an input query, measuring retrieval accuracy (e.g., top-1, top-3).
2.  **Augmentation & Generation:**
    *   The retrieved chunks (context) augment the original query.
    *   This combined input is fed into various language models (`code/libdocs/classifiers/`, including `acuopenai`, `anthropic`, `mistral`, `zephyr`, `deberta`) for generating context-aware outputs or "verdicts".
    *   The `llmchecker` module (`code/libdocs/llmchecker/`) may be used for evaluating or validating generated outputs.

## Fine-tuning ✨

The repository supports fine-tuning models, adapting them for improved performance on domain-specific tasks, with a specific focus on DeBERTa:

1.  **Dedicated Module:** `code/libdocs/finetune/` contains the core logic for managing and executing the fine-tuning process.
2.  **Model Focus:** Primarily targets the DeBERTa model architecture (`code/libdocs/classifiers/deberta/`).
3.  **Process:** Involves taking a pre-trained DeBERTa model and performing additional training using curated datasets (potentially derived from the processed PDFs or stored in `data/`). This tailors the model to the specific nuances of the project's data.
4.  **Evaluation:** Includes steps for rigorously testing the fine-tuned model and quantifying performance gains (e.g., accuracy metrics).
5.  **Utilities & Tracking:**
    *   Helper scripts in `code/utils/training/` likely support the workflow (data prep, training loops).
    *   Integration with `wandb` (`code/libdocs/wandb/`) enables robust experiment tracking and visualization during the fine-tuning process.

## Supporting Components 🛠️

*   **Metadata Handling:** `code/libdocs/metadata/` manages the `.metadata.json` files.
*   **Utilities:** `code/utils/` offers helpers for data cleaning (`data_cleaner`), GPU management (`gpu`), file handling (`jsonl`, `csv`), text processing (`text`), etc.
*   **API:** `code/libdocs/rest_api/` suggests a potential REST API endpoint for interaction.

## Summary 🎯

This repository defines a robust pipeline for ingesting PDFs, transforming them into indexed vector embeddings for effective Retrieval-Augmented Generation (RAG), and leveraging diverse LLMs—including specialized, fine-tuned models like DeBERTa—to perform sophisticated analysis and generate insights grounded in the document content.
