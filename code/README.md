
# Document Processing Pipeline

## Scripts

1. `file_sanity.py` ensures the naming on PDFs and contents of metadata are sane. It checks/fixes the following:
  - check file name consistency and if not rename them
  - check pdf file named as `name.pdf` has a metadata file named as `name.metadata.json`
  - check no arbitrary `.metadata.json` files exist
  - checks the contents of valid metadata files

2. `main.py` runs the pipline

3. `cleaner.py` will be renamed but this generates the model specific labels

4. `classifier_model.py` is finetuning of model, test and print accuracy

    - Must provide hf and wandb tokens to run
    - python code/classifier_model.py -t <hf-access-token> -w <wandb_access_token> ...

5. `classifier_knn.py` is knn classification based on input data.

    Examples:
    Do training and testing with default cleaned up files
    - `python code/classifier_knn.py`
    Do training and testing with complete data set
    - `python code/classifier_knn.py -f combined.jsonl -s label -t text`
