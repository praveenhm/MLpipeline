import logging

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def upload_csv_to_huggingface(
    csv_file_path, repo_name, token, file_name_in_repo, private=False
) -> str:
    """
    Uploads a CSV file to Hugging Face Datasets
    Args:
        csv_file_path: str: path to the CSV file
        repo_name: str: name of the repo in Hugging Face
        token: str: Hugging Face API token
        file_name_in_repo: str: name of the file in the Hugging Face dataset
        private: bool: whether the dataset is private
    Returns:
        str: message indicating the upload status
    """
    # Initialize Hugging Face API client
    api = HfApi()

    # Get the user name from the Hugging Face token
    full_repo_name = repo_name
    logging.info(f"Repo name: {full_repo_name}")

    api.create_repo(
        repo_id=full_repo_name,
        token=token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=csv_file_path,
        # path_in_repo=os.path.basename(csv_file_path),
        path_in_repo=file_name_in_repo,
        repo_id=full_repo_name,
        repo_type="dataset",
        token=token,
    )

    return f"Uploaded {file_name_in_repo} to {full_repo_name}"
