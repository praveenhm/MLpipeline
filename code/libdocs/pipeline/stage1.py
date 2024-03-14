import glob
import itertools
import logging
import os
import re
from typing import Any, Callable, List, Tuple, Type

from libdocs.chunker.basechunker import BaseChunker
from libdocs.google.bigquery import GoogleBigQueryProcessor
from libdocs.google.document import GooglePDFProcessor
from libdocs.google.storage import GoogleStorageProcessor
from libdocs.pipeline.utils import chunks_to_dataframe
from libdocs.splitter.splitter import Splitter
from typing_extensions import Self

control_chars = "".join(
    map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0)))
)
control_char_re = re.compile("[%s]" % re.escape(control_chars))


class PipelineObject:
    def __init__(
        self,
        path: str,
        split_dir=None,
        chunks_dir=None,
        gcs=None,
        gpp=None,
        gbq=None,
        chunker: Type[BaseChunker] = None,
        upload: bool = True,
        download: bool = False,
    ):
        if not os.path.exists(path):
            raise Exception(
                f"failed to create pipeline object: '{path}' does not exist"
            )
        if not path.endswith(".pdf"):
            raise Exception(
                f"failed to create pipeline object: '{path}' is not a supported document (PDF)"
            )
        if os.path.isdir(path):
            raise Exception(
                f"failed to create pipeline object: '{path}' is a directory"
            )
        if split_dir is None or not os.path.isdir(split_dir):
            raise Exception(
                f"failed to create pipeline object: split base dir '{split_dir}' does not exist or is not a directory"
            )
        if chunks_dir is None or not os.path.isdir(chunks_dir):
            raise Exception(
                f"failed to create pipeline object: chunks base dir '{chunks_dir}' does not exist or is not a direcotry"
            )
        if gcs is None or not isinstance(gcs, GoogleStorageProcessor):
            raise Exception(
                "failed to create pipeline object: gcs is not a GoogleStorageProcessor instance"
            )
        if gpp is None or not isinstance(gpp, GooglePDFProcessor):
            raise Exception(
                "failed to create pipeline object: gpp is not a GooglePDFProcessor instance"
            )
        if gbq is None or not isinstance(gbq, GoogleBigQueryProcessor):
            raise Exception(
                "failed to create pipeline object: gbq is not a GoogleBigQueryProcessor instance"
            )
        if chunker is None or not issubclass(chunker, BaseChunker):
            raise Exception(
                f"failed to create pipeline object: chunker {chunker} is not a subclass of BaseChunker"
            )
        self.pdf_file: str = os.path.basename(path)
        self.pdf_dir: str = os.path.dirname(path)
        self.topic: str = os.path.basename(self.pdf_dir)
        self.split_dir: str = os.path.join(split_dir, self.topic)
        os.makedirs(self.split_dir, exist_ok=True)
        self.split_files: List[str] = []
        self.chunks_dir: str = os.path.join(chunks_dir, self.topic)
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.dai_chunks_files: List[str] = []
        self.chunks_files: List[str] = []
        self.gcs: GoogleStorageProcessor = gcs
        self.gcs_uris: List[str] = []
        self.gcs_chunks_uris: List[str] = []
        self.gpp: GooglePDFProcessor = gpp
        self.gbq: GoogleBigQueryProcessor = gbq
        self.chunker: Type[BaseChunker] = chunker
        self.upload: bool = upload
        self.download: bool = download
        self.__exec_step_idx: int = 0
        self.__steps: List[Tuple[Callable[[Self], None], str, str]] = [
            (
                self.__getattribute__("splitter"),
                "Splitter",
                "Splits files in input-directory data/pdfs and makes it consumable for DocumentAI in directory data/pdfs/split",
            ),
            (
                self.__getattribute__("upload_split_files"),
                "Upload Split Files",
                "Uploads all split files to GCS so that they can be consumed by DocumentAI",
            ),
            (
                self.__getattribute__("documentai"),
                "DocumentAI",
                "Convert to text using DocumentAI which feeds all to GCS uploaded split files through DocumentAI",
            ),
            (
                self.__getattribute__("upload_documentai"),
                "Upload DocumentAI JSONL",
                "Uploading chunks from DocumentAI to GCS",
            ),
        ]

    def __repr__(self):
        return f"<PipelineObject {str(self.__dict__)}>"

    def __str__(self):
        return f"<PipelineObject {str(self.__dict__)}>"

    def step_num(self) -> int:
        """
        Returns the step number (NOT INDEX --> index+1) of the next step that is being executed when run_step() is called
        """
        if self.__exec_step_idx >= len(self.__steps):
            return 0
        return self.__exec_step_idx + 1

    def step_name(self) -> str:
        """
        Returns the name of the step of the next step that is being executed when run_step() is called
        """
        if self.__exec_step_idx >= len(self.__steps):
            return ""
        (_, name, _) = self.__steps[self.__exec_step_idx]
        return name

    def step_desc(self) -> str:
        """
        Returns the description of the step of the next step that is being executed when run_step() is called
        """
        if self.__exec_step_idx >= len(self.__steps):
            return ""
        (_, _, desc) = self.__steps[self.__exec_step_idx]
        return desc

    def is_finished(self) -> bool:
        """
        Returns true if all of the steps in the pipeline have been executed with run_step()
        """
        return self.__exec_step_idx >= len(self.__steps)

    def run_step(self):
        """
        Runs the next step if there are more steps to run in this pipeline, and increases the internal step counter
        """
        if self.__exec_step_idx >= len(self.__steps):
            return
        idx = self.__exec_step_idx
        # we increase the index before we execute because this might raise an exception, and we want to make sure
        # that we have increased the step counter anyways
        self.__exec_step_idx += 1
        (func, _, _) = self.__steps[idx]
        return func()

    def steps(self) -> List[Tuple[Any, str, str]]:
        return self.__steps

    def splitter(self):
        """
        Splits files in input-directory data/pdfs and makes it consumable for DocumentAI in directory data/pdfs/split
        """
        # step guard
        # condition 1: test if there are already split files, if there are, we assume that the split was successful
        # TODO: condition 2: test if there are already split files in the GCS bucket, if there are, we can download
        # them instead, and assume the split was successful
        filename = os.path.splitext(self.pdf_file)[0]
        split_files = glob.glob(f"{self.split_dir}/{filename}-*.pdf")
        if len(split_files) > 0:
            logging.info(
                f"Skipping splitter for {self.pdf_file}: already have {len(split_files)} splitted files"
            )
            self.split_files = split_files
            return

        # run step
        splitter = Splitter(self.pdf_dir, self.split_dir, [self.pdf_file])
        self.split_files = splitter.run()

    def upload_split_files(self):
        """
        Uploads all split files to GCS so that they can be consumed by DocumentAI
        """
        # step guard
        # condition: test if file is already uploaded, and if its MD5 sum matches
        # NOTE: step guard is part of the gcs.upload() method
        if self.upload:
            for split_file in self.split_files:
                gcs_uri = self.gcs.upload(split_file)
                self.gcs_uris.append(gcs_uri)
        else:
            for split_file in self.split_files:
                gcs_uri = self.gcs.gcs_uri_for_file(split_file)
                self.gcs_uris.append(gcs_uri)
                logging.info(f"Skipping upload of {split_file} to {gcs_uri}...")

    def documentai(self):
        """
        Convert to text using DocumentAI. Feeds all to GCS uploaded split files through DocumentAI
        """
        # step guard
        # condition 1: test if ".documentai.jsonl" file already exists locally
        # condition 2: test if ".documentai.jsonl" file already exists in the GCS bucket
        # if it does, then we're skipping the DocumentAI processing
        for gcs_uri in self.gcs_uris:
            # build path to output file path
            filename = gcs_uri.rsplit("/", 1)[-1]
            filename = os.path.splitext(filename)[0]
            filename = f"{filename}.documentai.jsonl"
            path = os.path.join(self.chunks_dir, filename)

            # step guard condition 1
            if os.path.exists(path):
                logging.info(
                    f"Skipping processing PDF at {gcs_uri} through DocumentAI: output file {path} already exists"
                )
                self.dai_chunks_files.append(path)
                continue

            # step guard condition 2 - if download is set
            if self.download:
                if self.gcs.exists_file(path, must_exist=False):
                    # download file as it already exists in the bucket
                    self.gcs.download_file(path)
                    self.dai_chunks_files.append(path)
                    logging.info(
                        f"Skipping processing PDF at {gcs_uri} through DocumentAI: file {path} already uploaded to GCS bucket"
                    )
                    continue

            # process in DocumentAI
            chunks = self.gpp.contents_for_gcs_uri(gcs_uri)
            # convert to DataFrame and JSON lines
            df = chunks_to_dataframe(chunks, self.topic, gcs_uri)
            json_lines = df.to_json(orient="records", lines=True)
            # write out to JSONL file
            with open(path, "w") as file:
                file.write(json_lines)
            self.dai_chunks_files.append(path)
            logging.info(
                f"DocumentAI processed PDF at {gcs_uri} and chunks written to {path}"
            )

    def upload_documentai(self):
        """
        Uploading chunks from DocumentAI to GCS
        """
        # step guard
        # condition: test if file is already uploaded, and if its MD5 sum matches
        # NOTE: step guard is part of the gcs.upload() method
        if self.upload:
            for chunks_file in self.dai_chunks_files:
                self.gcs.upload(chunks_file)
                # NOTE: do **not** append this URI to self.gcs_chunks_uris
        else:
            logging.info(
                f"Skipping upload of {len(self.dai_chunks_files)} DocumentAI JSONL files..."
            )

    def store_in_big_query(self):
        """
        Store in big query. Takes all the chunks as produced from DocumentAI and stores them in Google Big Query
        """
        for gcs_uri, gcs_chunks_uri in zip(self.gcs_uris, self.gcs_chunks_uris):
            self.gbq.delete_for_gcs_uri(gcs_uri)
            self.gbq.load_from_gcs_uri(gcs_chunks_uri)
            logging.info(
                f"Loaded chunks into Big Query table '{self.gbq.table_id}' for PDF at {gcs_uri} from JSONL at {gcs_chunks_uri}"
            )
