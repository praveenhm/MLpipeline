import datetime
import hashlib
import io
import logging
import os.path
import time
from typing import Any, Dict, Generator, Iterable, List, Union

from dateutil.parser import isoparse
from google.oauth2 import service_account
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from typing_extensions import Self

# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class GoogleDriveFile:
    """
    GoogleDriveFile represents a Google Drive File object.
    """

    def __init__(
        self,
        id: str,
        name: str,
        parent_id: str = "",
        mimeType: str = "",
        md5Checksum: str = "",
        modifiedTime: datetime.datetime = None,
        createdTime: datetime.datetime = None,
    ):
        if id is None or id == "":
            raise Exception("'id' must not be empty")
        if name is None or name == "":
            raise Exception("'name' must not be empty")
        if (
            modifiedTime is not None
            and not isinstance(modifiedTime, datetime.datetime)
            and isinstance(modifiedTime, str)
        ):
            modifiedTime = isoparse(modifiedTime)
        if (
            createdTime is not None
            and not isinstance(createdTime, datetime.datetime)
            and isinstance(createdTime, str)
        ):
            createdTime = isoparse(createdTime)
        self.id: str = id
        self.name: str = name
        self.parent_id: str = parent_id
        self.mimeType: str = mimeType
        self.md5Checksum: str = md5Checksum
        self.modifiedTime: datetime.datetime = modifiedTime
        self.createdTime: datetime.datetime = createdTime

    def __repr__(self):
        return f"<GoogleDriveFile {str(self.__dict__)}>"

    def __str__(self):
        return f"<GoogleDriveFile {str(self.__dict__)}>"

    @classmethod
    def from_get(cls, resp: Dict[str, Any]) -> Self:
        # there can be only one parent here, we know this
        # or even if there is not, we are fine with this (I guess)
        parent_id = ""
        parents = resp.get("parents", [])
        if len(parents) > 0:
            parent_id = parents[0]
        return cls(
            resp.get("id"),
            resp.get("name"),
            parent_id=parent_id,
            mimeType=resp.get("mimeType", None),
            md5Checksum=resp.get("md5Checksum", None),
            modifiedTime=resp.get("modifiedTime", None),
            createdTime=resp.get("createdTime", None),
        )

    @classmethod
    def from_list(cls, resp: Iterable[Dict[str, Any]]) -> List[Self]:
        return [cls.from_get(entry) for entry in resp]


class GoogleDriveProcessor:
    """
    GoogleDriveProcessor is capable of listing, downloading and watching our docs from a shared Google Drive folder.
    """

    def __init__(
        self,
        root_folder_id="159Ajqu1WPZtUQH9ikMQZFCQXgx9_lsEh",
        exclusion_names=["SV_subject_taxonomy"],
        service_account_credentials_file="service-account.json",
    ):
        """
        Construct a new 'GoogleDriveProcessor' object.

        :param root_folder_id: The root folder ID (e.g. 159Ajqu1WPZtUQH9ikMQZFCQXgx9_lsEh for "subject_taxonomy") where
            our taxonomy files are categorized and stored under.
        :param exclusion_names: All names that should be ignored when list_entries() is called
        :return: returns the GoogleDriveProcessor object
        """
        if root_folder_id == "":
            raise Exception("root_folder_id must not be empty")
        self.root_folder_id: str = root_folder_id
        self.exclusion_names: List[str] = exclusion_names
        if not os.path.isfile(service_account_credentials_file):
            raise Exception(
                f"service account credentials file at '{service_account_credentials_file}' does not exist or is not a file"
            )
        creds = service_account.Credentials.from_service_account_file(
            service_account_credentials_file
        )
        self.client: Resource = build("drive", "v3", credentials=creds)

    def list_folders(
        self,
        parent_id: str = None,
        after_datetime: datetime.datetime = None,
        exclusion_names: Iterable[str] = None,
    ) -> List[GoogleDriveFile]:
        """
        Lists all directories in a directory. It's a call to list_entries() with mime_type=FOLDER_MIME_TYPE.
        """
        return self.list_entries(
            parent_id, FOLDER_MIME_TYPE, after_datetime, exclusion_names
        )

    def list_entries(
        self,
        parent_id: str = None,
        mime_type: str = None,
        after_datetime: datetime.datetime = None,
        exclusion_names: Iterable[str] = None,
        query: str = None,
        recursive: bool = False,
    ) -> List[GoogleDriveFile]:
        """
        Lists all entries in a directory. If parent_id is empty or None, the root folder entries are listed.
        There are several filters that can be applied optionally:
        - only list entries which are created or modified after a certain date.
        - only list entries of a certain mime type
        - exclude names
        - any other query string

        :param parent_id: (optional) the ID of the parent folder if set, root folder of self otherwise
        :param mime_type: (optional) filter list by entries of this MIME type only
        :param after_datetime: (optional) only list files which were created or modified after a certain date
        :param exclusion_names: (optional) a list of names that must be excluded from the list if set, exclusion_names of self otherwise
        :param query: (optional) an additional query string to pass to the operation
        :return: returns a list of the files
        """
        folder_id: str = (
            self.root_folder_id
            if parent_id is None or parent_id == ""
            else parent_id
        )
        after_datetime_str: str = (
            ""
            if not isinstance(after_datetime, datetime.datetime)
            else f"and (modifiedTime >= '{after_datetime.isoformat(timespec='seconds')}' or createdTime >= '{after_datetime.isoformat(timespec='seconds')}')"
        )
        mimetype_str: str = (
            ""
            if mime_type is None or mime_type == ""
            else f"and mimeType = '{mime_type}'"
        )
        exclusion_str = (
            " ".join([f"and name != '{en}'" for en in self.exclusion_names])
            if not isinstance(exclusion_names, Iterable)
            else " ".join([f"and name != '{en}'" for en in exclusion_names])
        )
        additional_query_str = (
            "" if query is None or query == "" else f"and {query}"
        )
        querystr: str = (
            f"'{folder_id}' in parents {after_datetime_str} {mimetype_str} {exclusion_str} {additional_query_str}".strip()
        )
        # logging.info(f"{querystr=}")
        files: List[GoogleDriveFile] = []
        page_token = None
        while True:
            resp = (
                self.client.files()
                .list(
                    pageSize=20,
                    fields="nextPageToken, files(id, name, parents, mimeType, md5Checksum, modifiedTime, createdTime)",
                    corpora="user",
                    spaces="drive",
                    q=querystr,
                    pageToken=page_token,
                )
                .execute()
            )
            files.extend(GoogleDriveFile.from_list(resp.get("files", [])))
            page_token = resp.get("nextPageToken", None)
            if page_token is None:
                break
        if recursive:
            for file in files.copy():
                if file.mimeType == FOLDER_MIME_TYPE:
                    files.extend(
                        self.list_entries(
                            file.id,
                            mime_type,
                            after_datetime,
                            exclusion_names,
                            query,
                            recursive,
                        )
                    )
        return files

    def exists(self, file_id: str) -> bool:
        """
        Returns true if the object for the given file ID exists.
        NOTE: it is recommended to use get() over exists, as it returns a GoogleDriveFile object in case it does.
        """
        return self.get(file_id) is not None

    def get(self, file_id: str) -> Union[GoogleDriveFile, None]:
        """
        Returns a GoogleDriveFile object for the given file ID, or None if the object does not exist.
        Any other error raises exceptions.
        """
        try:
            resp: Dict[str, Any] = (
                self.client.files()
                .get(
                    fileId=file_id,
                    fields="id, name, parents, mimeType, md5Checksum, modifiedTime, createdTime",
                )
                .execute()
            )
            return GoogleDriveFile.from_get(resp)
        except HttpError as err:
            if err.status_code == 404:
                return None
            else:
                raise

    def get_folder(
        self, folder_name: str, parent_id: str = None
    ) -> Union[GoogleDriveFile, None]:
        """
        Gets the folder with the specified name within the parent folder. If parent ID is not set, it looks at the root folder.
        """
        return self.get_file(folder_name, parent_id, FOLDER_MIME_TYPE)

    def get_file(
        self, file_name: str, parent_id: str = None, mime_type: str = None
    ) -> Union[GoogleDriveFile, None]:
        """
        Gets the file with the specified name within the parent folder with the given ID. If parent ID is not set, it looks at the root folder.
        """
        for file in self.list_entries(
            parent_id, mime_type=mime_type, query=f"name = '{file_name}'"
        ):
            if file.name == file_name:
                return file
        return None

    def download_file_from_subject(
        self,
        subject_name: str,
        file_name: str,
        dir: str = None,
        stream: bool = False,
    ) -> Union[bytes, None]:
        """
        Downloads the file with file_name from one of the top-level folders with subject_name.
        The other optional parameters are ths ame as for download_id().
        """
        obj = self.get_folder(subject_name)
        if obj is None:
            raise Exception(
                f"folder with name '{subject_name}' was not found in the root folder with ID '{self.root_folder_id}'"
            )
        return self.download_file(obj.id, file_name, dir, stream)

    def download_file(
        self,
        parent_id: str,
        file_name: str,
        dir: str = None,
        stream: bool = False,
    ) -> Union[bytes, None]:
        """
        Downloads the file with name form the folder with the given ID from Google Drive.
        The other optional parameters are the same as for download_id().
        """
        obj = self.get_file(file_name, parent_id)
        if obj is None:
            raise Exception(
                f"file with name '{file_name}' was not found in folder with ID '{parent_id}'"
            )
        return self.download_id(obj.id, dir, file_name, stream)

    def download_id(
        self,
        file_id: str,
        dir: str = None,
        dest_file_name: str = None,
        stream: bool = False,
    ) -> Union[bytes, None]:
        """
        Downloads the file with the ID from Google Drive.
        If a dir is given the file is being downloaded to the target directory, otherwise it
        is being saved to the current working directory. If stream is True, it will not
        save a

        :param parent_id: the file ID of the parent directory holding the file
        :param file_id: the file ID to download from the folder with parent ID
        :param dir: (optional) download destination directory
        :param dest_file_name: (optional) name of the destination file:
        :param stream: (optional) if True, will return bytes instead of saving to a file
        :return: the bytes of the file if stream is True, or None otherwise
        """
        if file_id == "":
            raise Exception("file_id must not be empty")
        obj: GoogleDriveFile = self.get(file_id)
        if obj is None:
            raise Exception(f"object with file_id '{file_id}' does not exist")
        request = self.client.files().get_media(fileId=file_id)
        fh: io.BytesIO = None
        out_path: str = ""
        if stream:
            fh = io.BytesIO()
        else:
            dest_dir = os.getcwd() if dir is None or dir == "" else dir
            if not os.path.isdir(dest_dir):
                raise Exception(
                    f"{dest_dir} does not exist or is not a directory"
                )
            dest_file = (
                file_id
                if dest_file_name is None or dest_file_name == ""
                else dest_file_name
            )
            out_path = os.path.join(dest_dir, dest_file)
            if os.path.exists(out_path) and obj.md5Checksum != "":
                dest_file_md5 = md5_hexdigest_for_file(out_path)
                if dest_file_md5 == obj.md5Checksum:
                    logging.info(
                        f"Skipping download of '{file_id}' to {out_path}: file already exists, and MD5 sums match"
                    )
                    return None
            fh = io.FileIO(out_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logging.debug("Download %d%%." % int(status.progress() * 100))
        if stream:
            return fh.getvalue()
        else:
            logging.info(f"Downloaded '{file_id}' to {out_path}")
            return None

    def delete_file(self, file_name: str, parent_id: str = None):
        """
        Deletes a file with the given name from Google Drive. If parent ID is not set, it will assume the root folder.
        """
        file_id = self.get_file(file_name, parent_id)
        return self.delete(file_id)

    def delete(self, file_id: str):
        """
        Deletes a file with the given ID.
        """
        if file_id == "":
            raise Exception("file_id must not be empty")
        self.client.files().delete(fileId=file_id).execute()

    def upload_file(
        self,
        path: str,
        parent_id: str = None,
    ) -> str:
        # input checks
        if not os.path.isfile(path):
            raise Exception(f"{path} does not exist or is not a file")
        pid = (
            parent_id
            if parent_id is not None and parent_id != ""
            else self.root_folder_id
        )
        file_name = os.path.basename(path)
        obj = self.get_file(file_name, pid)
        if obj is None:
            # create it
            body = {"name": file_name, "parents": [pid]}
            media_body = MediaFileUpload(path)
            file = (
                self.client.files()
                .create(
                    body=body,
                    media_body=media_body,
                    fields="id, name, parents",
                    enforceSingleParent=True,
                )
                .execute()
            )
            logging.info(f"Uploaded {path} to folder '{pid}' [create]")
            return file.get("id")
        else:
            # check if the MD5 sums match first
            src_md5 = md5_hexdigest_for_file(path)
            if obj.md5Checksum == src_md5:
                logging.info(
                    f"Skipping upload of {path}: file already exists in '{pid}', and MD5 sums match"
                )
                return obj.id
            # update the existing file
            media_body = MediaFileUpload(path)
            file = (
                self.client.files()
                .update(
                    fileId=obj.id,
                    media_body=media_body,
                    fields="id",
                    enforceSingleParent=True,
                )
                .execute()
            )
            logging.info(f"Uploaded {path} to folder '{pid}' [update]")
            return file.get("id")

    def watch(
        self,
        folder_id: str = None,
        interval_secs: float = 15,
        recursive: bool = False,
    ) -> Generator[List[GoogleDriveFile], None, None]:
        fid = (
            folder_id
            if folder_id is not None and folder_id != ""
            else self.root_folder_id
        )
        dt: datetime.datetime = None
        while True:
            time.sleep(interval_secs)
            dt1 = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
            if dt is None:
                dt = dt1
            logging.info(
                f"Watch cycle (now {dt1.isoformat(timespec='seconds')})  dateTime >= {dt.isoformat(timespec='seconds')}"
            )
            entries = self.list_entries(
                fid, after_datetime=dt, recursive=recursive
            )
            dt = dt1
            if len(entries) > 0:
                yield entries


def md5_hexdigest_for_file(path: str) -> str:
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


if __name__ == "__main__":
    p = GoogleDriveProcessor()
    logging.basicConfig(level=logging.INFO)
    entries = p.list_entries(recursive=True)
    for entry in entries:
        print(
            f"{entry.id=} {entry.name=} m={entry.modifiedTime.isoformat()} c={entry.createdTime.isoformat()}"
        )
    logging.info("watcher")
    w = p.watch(recursive=True)
    for nes in w:
        logging.info(f"Received new entries (total {len(nes)})")
        for obj in nes:
            logging.info(
                f"{obj.id=} {obj.name=} m={obj.modifiedTime.isoformat()} c={obj.createdTime.isoformat()}"
            )
