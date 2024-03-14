import base64
import hashlib
import logging
import os
import re
from typing import List

from google.cloud import storage

gcs_uri_pattern = r"gs://([^/]+)/(.+)"


class GoogleStorageProcessor:
    """
    GoogleStorageProcessor is capable of reading, downloading and uploading docs from GCS buckets.
    """

    def __init__(self, bucket="docprocessor", base_dir=""):
        """
        Construct a new 'GoogleStorageProcessor' object.

        :param bucket: The bucket name on GCS where files are stored.
        :param base_dir: The local directory base path to use when paths are uploaded or downloaded.
        :return: returns nothing
        """
        if base_dir == "":
            self.base_dir = os.getcwd()
        elif not os.path.isdir(base_dir):
            raise Exception(f"{base_dir=} does not exist or is not a directory")
        else:
            self.base_dir = base_dir
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket)

    def files(self, directory=None) -> List[str]:
        """
        Provide the files in the bucket. You can optionally list only the ones from a sub directory by providing a directory as a prefix filter.
        :param directory: sub directory filter
        :return: returns list of files.
        """
        blobs = self.bucket.list_blobs(prefix=directory)
        blob_names = {}
        for b in blobs:
            blob_names[b.name] = b
        return self.blob_names.keys()

    def blobs(self, directory=None) -> List[storage.Blob]:
        """
        Provides the blobs in the bucket. You can optionally list only the ones from a sub directory by providing a directory as a prefix filter.
        :param directory: sub directory filter
        :return: returns list of blobs.
        """
        blobs = self.bucket.list_blobs(prefix=directory)
        return [blob for blob in blobs]

    def _path_from_uri(self, gcs_uri: str) -> str:
        """
        Returns the path from a GCS URI. This will raise an exception if the bucket in the URI does not match the bucket of self, or if this is not a GCS URI.
        """
        # extract bucket and path, and validate input
        bucket_name = ""
        path = ""
        match = re.match(gcs_uri_pattern, gcs_uri)
        if match:
            bucket_name = match.group(1)
            path = match.group(2)
        else:
            raise Exception("'{gcs_uri}' is not a valid GCS URI")
        if bucket_name != self.bucket.name:
            raise Exception(
                "buckets do not match: gcs_uri bucket: '{}' != self.bucket.name '{self.bucket.name}'"
            )
        return path

    def _path_from_filepath(self, filepath: str, base_dir="") -> str:
        """
        Returns the relative bucket path from a file at filepath. Filepath is a path to a file on the filesystem, and the file must exist.
        The bucket path tested will be relative to self.base_dir if not overridden with base_dir in this function.
        The base directory must also refer to a parent directory of the filepath.
        """
        if not os.path.isfile(filepath):
            raise Exception(f"{filepath} does not exist or is not a file")
        dir = base_dir if base_dir != "" else self.base_dir
        if not os.path.isdir(dir):
            raise Exception(f"{dir} does not exist or is not a directory")

        abs_filepath = os.path.abspath(filepath)
        abs_basedir = os.path.abspath(dir)
        if not abs_basedir.endswith("/"):
            abs_basedir += "/"
        if not abs_filepath.startswith(abs_basedir):
            raise Exception(
                f"{dir} is not a parent directory of input filepath {filepath}"
            )
        path = abs_filepath.removeprefix(abs_basedir)
        return path

    def _path_from_possible_filepath(self, filepath: str, base_dir="") -> str:
        path = ""
        dir = base_dir if base_dir != "" else self.base_dir
        if not os.path.isdir(dir):
            raise Exception(f"{dir} does not exist or is not a directory")
        abs_basedir = os.path.abspath(dir)
        if not abs_basedir.endswith("/"):
            abs_basedir += "/"
        if filepath.startswith("/"):
            if not filepath.startswith(abs_basedir):
                raise Exception(
                    f"{dir} is not a parent directory of {filepath}"
                )
            path = filepath.removeprefix(abs_basedir)
        elif filepath.startswith("./"):
            abs_path = os.path.join(os.getcwd(), filepath.removeprefix("./"))
            if not abs_path.startswith(abs_basedir):
                raise Exception(
                    f"{dir} is not a parent directory of {filepath}"
                )
            path = abs_path.removeprefix(abs_basedir)
        else:
            path = filepath
        return path

    def gcs_uri_for_file(
        self, filepath: str, base_dir: str = "", must_exist: bool = True
    ) -> str:
        """
        Returns a GCS URI for the file at filepath. Filepath is a path to a file on the filesystem, and the file must exist, unless must_exist is False.
        The bucket path within the GCS URI will be relative to self.base_dir if not overridden with base_dir in this function. The base directory must
        also refer to a parent directory of the filepath.

        :param filepath: path to a file on the filesystem for which to create a GCS URI
        :param base_dir: (optional) path to a parent directory of filepath which serves as the base_dir of the upload directory
        :return: a well formatted GCS URI
        """
        path: str = ""
        path = (
            self._path_from_filepath(filepath, base_dir)
            if must_exist
            else self._path_from_possible_filepath(filepath, base_dir)
        )
        return f"gs://{self.bucket.name}/{path}"

    def exists_uri(self, gcs_uri: str) -> bool:
        """
        Checks if the blob with the GCS URI exists. This will raise an exception if the bucket in the URI
        does not match the bucket of self, or if this is not a GCS URI.

        :return: true if the blob exists
        """
        path = self._path_from_uri(gcs_uri)
        blob = self.bucket.blob(path)
        return blob.exists()

    def exists_file(
        self, filepath: str, base_dir="", check_md5=True, must_exist=True
    ) -> bool:
        """
        Checks if the blob at filepath exists. Filepath is a path to a file on the filesystem, and the file must exist, unless must_exist is False.
        The bucket path tested will be relative to self.base_dir if not overridden with base_dir in this function.
        The base directory must also refer to a parent directory of the filepath.

        :param filepath: path to a file on the filesystem to check against the GCS bucket
        :param base_dir: (optional) path to a parent directory of filepath which serves as the base_dir of the upload directory
        :param check_md5: (optional) if True, will additionally check if the MD5 sum of an existing path in the bucket matches if the file exists locally
        """
        # get the bucket path first
        path = (
            self._path_from_filepath(filepath, base_dir)
            if must_exist
            else self._path_from_possible_filepath(filepath, base_dir)
        )

        # check if the blob already exists, and if it does, see if we need to reupload by checking against the MD5 checksum
        blob = self.bucket.blob(path)
        if blob.exists():
            # if check_md5 isn't requested, then we simply consider this to be enough of a check
            if not check_md5:
                return True

            # blob reload is necessary to retrieve the properties of the blob
            if os.path.exists(filepath):
                blob.reload()
                md5_src = md5_digest_for_file(filepath)
                return md5_src == blob.md5_hash
            else:
                # we can't check against the local file as we don't have it
                return True
        return False

    def exists(self, path: str) -> bool:
        """
        Checks if the blob at path exists. The path is relative to the bucket of self.

        :return: true if the blob exists
        """
        blob = self.bucket.blob(path)
        return blob.exists()

    def upload(self, filepath: str, base_dir="", check_md5=True) -> str:
        """
        Uploads a file to the Google Cloud Storage bucket. The directory structure of path will be preserved during upload.
        The path will be relative to self.base_dir if not overridden with base_dir in this function.

        :param filepath: the path of the blob to upload. This should be relative to the upload destination path within the bucket
        :param base_dir: the path will be relative to this base directory. If not set, the base_dir of self will be used
        :return: the GCS URI of the uploaded blob
        """
        # get the bucket path first
        path = self._path_from_filepath(filepath, base_dir)
        gcs_uri = f"gs://{self.bucket.name}/{path}"

        # check if the blob already exists, and if it does, see if we need to reupload by checking against the MD5 checksum
        blob = self.bucket.blob(path)
        if blob.exists():
            # if check_md5 isn't requested, then we simply consider this to be enough of a check
            if not check_md5:
                logging.info(f"Skipping upload for {path}: blob already exists")
                return gcs_uri

            # blob reload is necessary to retrieve the properties of the blob
            blob.reload()
            md5_src = md5_digest_for_file(filepath)
            if md5_src == blob.md5_hash:
                logging.info(
                    f"Skipping upload for {path}: blob already exists, and MD5 sums match"
                )
                return gcs_uri

        # if it doesn't exist yet or MD5 sums don't match, then we are going to upload the file
        blob.upload_from_filename(filepath)
        logging.info(f"Uploaded to {gcs_uri}")
        return gcs_uri

    def data(self, path: str) -> bytes:
        """
        Provide the contents of a file.

        :return: returns list of paragraphs.
        """
        blob = self.bucket.blob(path)
        return blob.download_as_bytes()

    def download_file(self, filepath: str, base_dir=""):
        path = self._path_from_possible_filepath(filepath, base_dir)
        return self.download(path, base_dir)

    def download(self, path: str, base_dir=""):
        """
        Downloads the contents of a file at path in the bucket to a local file at the same relative path.
        The destination is relative to self.base_dir if not overriden with base_dir in this function.
        """
        dir = base_dir if base_dir != "" else self.base_dir
        destfile = os.path.join(dir, path)
        os.makedirs(os.path.dirname(destfile), exist_ok=True)
        blob = self.bucket.blob(path)
        if os.path.exists(destfile):
            blob.reload()
            md5_dest = md5_digest_for_file(destfile)
            if md5_dest == blob.md5_hash:
                logging.info(
                    f"Skipping download for {path}: blob already exists at {destfile}, and MD5 sums match"
                )
                return
        blob.download_to_filename(destfile)
        logging.info(f"Downloaded to {destfile}")

    def download_uri(self, gcs_uri: str, base_dir=""):
        """
        Downloads the contents of a file at GCS URI to a local file at the same relative path.
        This will raise an exception if the bucket in the URI does not match the bucket of self, or if this is not a GCS URI.
        The destination is relative to self.base_dir if not overridden with base_dir in this function.
        """
        path = self._path_from_uri(gcs_uri)
        dir = base_dir if base_dir != "" else self.base_dir
        destfile = os.path.join(dir, path)
        os.makedirs(os.path.dirname(destfile), exist_ok=True)
        blob = self.bucket.blob(path)
        if os.path.exists(destfile):
            blob.reload()
            md5_dest = md5_digest_for_file(destfile)
            if md5_dest == blob.md5_hash:
                logging.info(
                    f"Skipping download for {path}: blob already exists at {destfile}, and MD5 sums match"
                )
                return
        blob.download_to_filename(destfile)
        logging.info(f"Downloaded to {destfile}")


def md5_digest_for_file(path: str) -> str:
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return base64.b64encode(file_hash.digest()).decode("ascii")
