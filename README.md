# Background

Drop PDFs here and ensure you have a meta data file.


# Pipeline

## Stage 0 - Retrieve Input

Can be executed with defaults by running `./stage0`.
Its source begins in `code/downloader.py`.

**Input:** Google Drive folder

**Output:** PDFs and `.metadata.json` files

- [x] Downloads PDFs from Google Drive
  - downloads them from the `subject_taxonomy` Google Drive
  - only downloads if the file does not already exist and its MD5 sum matches
- [x] Runs file sanity on them
    - renames the files if necessary
    - ensures that a `.pdf` and `.metadata.json` file exist per PDF
    - instantiates `Metadata` to make sure they work

## Stage 1 - Initial Data Ingestion

Can be executed with defaults by running `./stage1`.
Its source beings in `code/ingestion.py`.

**Input:** PDFs and `.metadata.json` files
**Output:** `.documentai.jsonl` files, , and `documentai.combined.{jsonl,parquet,html}`

- [x] Splitter.
  --pdf data/pdfs/biz_dev/one.pdf --pdf data/pdfs/cybersecurity/*
  --split-dir data/processing/pdfs/split/
  Splits files in input-directory data/pdfs and makes it consumable for documentAI in directory data/pdfs/split.

- [x] Upload split files to GCS.
  --split-dir data/pdfs/split/
  --gcs-bucket docprocessor
  --gcs-base-dir data/processing/

- [x] Convert to text using DocumentAI
  Cleanup as much as possible
  --chunks-dir data/processing/pdfs/chunks/
  --gcp-project-id development-398309
  --gcp-dai-location us
  --gcp-dai-processor-id 277f11647ef22bec

- [x] Upload chunks from DocumentAI to GCS
  --chunks-dir data/processing/pdfs/chunks/
  --gcs-bucket docprocessor
  --gcs-base-dir data/processing/

## Stage 2 - Chunker

Can be executed with defaults by running `./stage2`.
Its source beings in `code/chunker.py` (`code/libdocs/chunker/`).

**Input:** `.documentai.jsonl` files (Text chunks from DocumentAI)
**Output:** `.batchchunker.jsonl` files (Refined, potentially semantic chunks), and combined `batchchunker.combined.{jsonl,parquet,html}`

- [x] 📄➡️🧩 Process chunks from DocumentAI through our custom chunker (`code/libdocs/chunker/`) for potentially improved semantic coherence or task-specific segmentation.
  --chunks-dir data/processing/pdfs/chunks/

- [x] ☁️ Upload enhanced chunks from the custom Chunker to GCS.
  --chunks-dir data/processing/pdfs/chunks/
  --gcs-bucket docprocessor
  --gcs-base-dir data/processing/

## Stage 3 - Embedding, Indexing, RAG & Fine-tuning 🧠🔍⚙️

This stage focuses on leveraging the processed text chunks for advanced analysis using Retrieval-Augmented Generation (RAG) and fine-tuned models.

**Input:** `.batchchunker.jsonl` files (Refined chunks)
**Output:** Model verdicts, evaluation metrics, indexed data.

- [x] 💾 **Store & Prepare:** Persist refined chunks from GCS into BigQuery for structured access.
  --gcs-bucket docprocessor
  --gcs-base-dir data/processing/
  --gcp-bq-dataset training
  --gcp-bq-table pdfs

- [ ] 🔢 **Embed:** Generate vector embeddings for each chunk using models specified in `code/libdocs/embedder/` (potentially leveraging `code/libdocs/huggingface/` or cloud provider APIs). These embeddings capture the semantic meaning of the text.

- [ ] 🗂️ **Index:** Build a high-performance vector index (e.g., using FAISS via `code/libdocs/faissindexer/`) from the generated embeddings. This enables ultra-fast similarity searches across the entire document corpus.

- [ ] 🔍 **Retrieval (RAG - Part 1):** Given a query, perform a vector search against the index to retrieve the most relevant document chunks (top-k). This forms the "augmented context".
    - Evaluate retrieval performance (e.g., top-1, top-3 accuracy).

- [ ] 🤖 **Generation & Analysis (RAG - Part 2 & Model Verdicts):** Feed the retrieved context along with the original query into various LLMs (`code/libdocs/classifiers/`) to generate informed "verdicts" or answers. Supported models include:
    - OpenAI (`acuopenai`)
    - Anthropic (`anthropic`)
    - Cohere (Implied, check `classifiers`)
    - Mistral (`mistral`)
    - Zephyr (`zephyr`)
    - Fine-tuned models (see below)
    - Use `code/libdocs/llmchecker/` for potential output validation.
  --gcp-bq-dataset training
  --gcp-bq-table pdfs

- [ ] ✨ **Fine-tuning (Focus: DeBERTa):** Improve model performance on domain-specific tasks by fine-tuning pre-trained models.
    - **Module:** Utilizes `code/libdocs/finetune/` and potentially helpers in `code/utils/training/`.
    - **Target Model:** Primarily focused on DeBERTa (`code/libdocs/classifiers/deberta/`).
    - **Process:** Further train the model on curated datasets (potentially derived from the ingested PDFs or stored in `data/`).
    - **Tracking:** Leverage `wandb` (`code/libdocs/wandb/`) for experiment tracking and visualization.
    - **Evaluation:** Test the fine-tuned model rigorously and measure accuracy improvements.

- [ ] 📊 **Visualization (Optional):** Analyze embedding space using techniques like UMAP (potentially via `code/libdocs/google/` or other plotting libs) for insights into data clusters and relationships.

**Overall Goal:** To create a robust system capable of understanding and reasoning over large PDF document sets by combining efficient retrieval with the power of large language models, including specialized, fine-tuned versions.
