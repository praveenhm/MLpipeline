
# Background

Drop PDFs here and ensure you have a meta data file.

Filenames must be lowercase and underscores. Convert spaces, +, - to _.
Create a metadata.json which for a file like a.pdf is a.metadata.json.

https://drive.google.com/drive/folders/159Ajqu1WPZtUQH9ikMQZFCQXgx9_lsEh



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
Its source beings in `code/chunker.py`.

**Input:** `.documentai.jsonl` files
**Output:** `.batchchunker.jsonl` files, and `batchchunker.combined.{jsonl,parquet,html}`

- [x] Process chunks from DocumentAI through our additional chunker
  --chunks-dir data/processing/pdfs/chunks/

- [x] Upload chunks from Chuker to GCS
  --chunks-dir data/processing/pdfs/chunks/
  --gcs-bucket docprocessor
  --gcs-base-dir data/processing/

## Stage 3 - Model Verdicts

- [x] Step 8: Store in big query
  --gcs-bucket docprocessor
  --gcs-base-dir data/processing/
  --gcp-bq-dataset training
  --gcp-bq-table pdfs

- [ ] Step 9: Add verdicts from ML models
  - OpenAI
  - Anthropic
  - Chohere
  - Deberta
  --gcp-bq-dataset training
  --gcp-bq-table pdfs

- [ ] Step 10:
  - given a dataset
    - vector-search
      - give me the top1,2,3 accuracy
    - deberta fine tuning
      - fine-tuning
      - test and give the accuracy

- [ ] Step 11: Umap Analyzer for Google
