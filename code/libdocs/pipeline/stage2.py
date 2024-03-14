import itertools
import logging
import os
import re
from typing import Any, Callable, List, Tuple, Type

from libdocs.chunker.basechunker import BaseChunker
from libdocs.google.storage import GoogleStorageProcessor
from libdocs.pipeline.utils import chunks_to_dataframe
from libdocs.utils.jsonl.jsonl import JSONL
from typing_extensions import Self

control_chars = "".join(
    map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0)))
)
control_char_re = re.compile("[%s]" % re.escape(control_chars))


class PipelineObject:
    def __init__(
        self,
        dai_chunks_files: List[str],
        chunks_dir=None,
        gcs=None,
        chunker: Type[BaseChunker] = None,
        upload: bool = True,
        download: bool = False,
    ):
        if len(dai_chunks_files) == 0:
            raise Exception(
                "failed to create pipeline object: dai_chunks_files is empty"
            )
        if chunks_dir is None or not os.path.isdir(chunks_dir):
            raise Exception(
                f"failed to create pipeline object: chunks base dir '{chunks_dir}' does not exist or is not a direcotry"
            )
        if gcs is None or not isinstance(gcs, GoogleStorageProcessor):
            raise Exception(
                "failed to create pipeline object: gcs is not a GoogleStorageProcessor instance"
            )
        if chunker is None or not issubclass(chunker, BaseChunker):
            raise Exception(
                f"failed to create pipeline object: chunker {chunker} is not a subclass of BaseChunker"
            )
        for dai_chunks_file in dai_chunks_files:
            if not os.path.isfile(dai_chunks_file):
                raise Exception(
                    f"failed to create pipeline object: dai_chunks_files: {dai_chunks_file} does not exist or is not a file"
                )
            if not dai_chunks_file.endswith(".documentai.jsonl"):
                raise Exception(
                    f"failed to create pipeline object: dai_chunks_files: {dai_chunks_file} does not end with '.documentai.jsonl'"
                )
        self.chunks_dir: str = os.path.abspath(chunks_dir)
        self.dai_chunks_files: List[str] = dai_chunks_files
        self.chunks_files: List[str] = []
        self.gcs: GoogleStorageProcessor = gcs
        self.gcs_chunks_uris: List[str] = []
        self.chunker: Type[BaseChunker] = chunker
        self.upload: bool = upload
        self.download: bool = download
        self.__exec_step_idx: int = 0
        self.__steps: List[Tuple[Callable[[Self], None], str, str]] = [
            (
                self.__getattribute__("run_chunker"),
                "Chunker",
                "Process chunks from DocumentAI through our additional chunker",
            ),
            (
                self.__getattribute__("upload_chunks"),
                "Upload Chunker JSONL",
                "Uploading chunks from Chunker to GCS",
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

    def run_chunker(self) -> List[Tuple[str, Exception]]:
        """
        Process chunks from DocumentAI through our additional chunker
        """
        # step guard
        # condition 1: test if ".jsonl" file already exists locally
        # condition 2: test if ".jsonl" file already exists in the GCS bucket
        # if it does, then we're skipping the DocumentAI processing
        chunker = self.chunker()
        errors: List[Tuple[str, Exception]] = []
        for dai_chunks_file in self.dai_chunks_files:
            try:
                # build path to output file path, and do some checks before hand
                if not os.path.isfile(dai_chunks_file):
                    logging.warning(
                        f"Skipping run_chunker for {dai_chunks_file}: {dai_chunks_file} does not exist or is not a file"
                    )
                    continue
                if not dai_chunks_file.endswith(".documentai.jsonl"):
                    logging.warning(
                        f"Skipping run_chunker for {dai_chunks_file}: file name does not end with '.documentai.jsonl'"
                    )
                    continue

                # the DocumentAI chunk file must be in the ordered directory structure
                abs_dai = os.path.abspath(dai_chunks_file)
                if not abs_dai.startswith(self.chunks_dir):
                    raise Exception(
                        f"chunks_dir {self.chunks_dir} is not part of the directory tree of {dai_chunks_file}"
                    )

                # because we determine the topic based on that, as well as the final destination of the file
                topic = os.path.basename(os.path.dirname(dai_chunks_file))
                filename = os.path.basename(dai_chunks_file).removesuffix(
                    ".documentai.jsonl"
                )
                filename = f"{filename}.{self.chunker.chunker_name()}.jsonl"
                path = os.path.join(self.chunks_dir, topic, filename)

                # step guard condition 1
                if os.path.exists(path):
                    logging.info(
                        f"Skipping processing DocumentAI JSONL at {dai_chunks_file} through Chunker: output file {path} already exists"
                    )
                    self.chunks_files.append(path)
                    continue

                # step guard condition 2 - only if download has been requested
                if self.download:
                    if self.gcs.exists_file(path, must_exist=False):
                        # download file as it already exists in the bucket
                        self.gcs.download_file(path)
                        self.chunks_files.append(path)
                        logging.info(
                            f"Skipping processing DocumentAI JSONL at {dai_chunks_file} through Chunker: file {path} already uploaded to GCS bucket"
                        )
                        continue

                # read the .documentai.jsonl files again as chunks
                jl = JSONL()
                jl.from_files(
                    os.path.dirname(dai_chunks_file),
                    os.path.basename(dai_chunks_file),
                )
                chunks = jl.df["text"].tolist()

                # clean them further
                cleaned_chunks = clean_chunks(chunks)

                # run through chunker model
                text = "\n".join(cleaned_chunks)
                new_chunks = chunker.process_text(text)
                # convert to DataFrame and JSON lines
                df = chunks_to_dataframe(
                    new_chunks,
                    topic,
                    self.gcs.gcs_uri_for_file(dai_chunks_file),
                )
                json_lines = df.to_json(orient="records", lines=True)
                # write out to JSONL file
                with open(path, "w") as file:
                    file.write(json_lines)
                self.chunks_files.append(path)
                logging.info(
                    f"Chunker processed JSONL file {dai_chunks_file} and chunks written to {path}"
                )
            except Exception as err:
                errors.append((dai_chunks_file, err))
        return errors

    def upload_chunks(self):
        """
        Uploading chunks from Chunker to GCS
        """
        # step guard
        # condition: test if file is already uploaded, and if its MD5 sum matches
        # NOTE: step guard is part of the gcs.upload() method
        if self.upload:
            for chunks_file in self.chunks_files:
                gcs_uri = self.gcs.upload(chunks_file)
                self.gcs_chunks_uris.append(gcs_uri)
        else:
            for chunks_file in self.chunks_files:
                gcs_uri = self.gcs.gcs_uri_for_file(chunks_file)
                self.gcs_chunks_uris.append(gcs_uri)
                logging.info(
                    f"Skipping upload of {chunks_file} to {gcs_uri}..."
                )


def clean_chunks(chunks: List[str]) -> List[str]:
    cleaned_chunks: List[str] = []
    for sentence in chunks:
        cleaned = control_char_re.sub("", sentence)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Remove lines that are numbers or number.number
        cleaned = re.sub(r"^[0-9\.\s]+$", " ", cleaned)
        # Multi-lines to space
        cleaned = re.sub(r"\n+", " ", cleaned).strip()
        # Multi-space to single space
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 25:
            continue
        cleaned_chunks.append(cleaned) if (len(cleaned) > 0) else None
    return cleaned_chunks
