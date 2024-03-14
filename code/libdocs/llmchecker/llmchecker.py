import logging
import os
import time
from typing import List

from libdocs.classifiers.common.types import ClassificationRequest
from libdocs.types.types import ChunkSubject
from libdocs.utils.banner.banner import banner
from libdocs.utils.jsonl.jsonl import JSONL

DATA_CORRECT_STR = "correct_data"
DATA_PARTLY_CORRECT_STR = "partly_correct_data"
DATA_INCORRECT_STR = "incorrect_data"


class LLMChecker:
    """
    LLMChecker is a class that allows us to check the labels from the LLMs and put them in three buckets:
        - correct           : if the input subject on the chunk matches the best match from model
        - partially correct : if the input subject on the chunk is contained in the top matches from the model
        - incorrect         : if the input subject does not match the matches provided by the model
    """

    def __init__(self):
        self.model = None

        # counters
        self.count_correct = 0
        self.count_partial = 0
        self.count_incorrect = 0

        # data
        self.data_correct = []
        self.data_partial = []
        self.data_incorret = []

    def __persist_as_jsonl(
        self, chunks: List[ChunkSubject], file_name: str, suffix: str
    ) -> None:
        """
        Persist the chunk to the file as jsonl

        :param chunks: the chunks to be persisted.
        :return: None
        """
        # pre-conditions check
        full_filename = file_name + "_" + suffix
        if not chunks or len(chunks) == 0:
            logging.info(
                f"Not Persisting the {len(chunks)}-chunks to the file: {full_filename}"
            )
            return

        logging.info(
            f"Persisting the {len(chunks)}-chunks to the file: {full_filename}"
        )
        with open(full_filename, "w") as file:
            for chunk in chunks:
                file.write(chunk.json() + "\n")
        logging.info(
            f"Persisted the {len(chunks)}-chunks to the file: {full_filename}"
        )

    def __flush(self, filebase: str):
        """
        Flush the batch

        :param filebase: the prefix of the file.
        """
        total = self.count_correct + self.count_partial + self.count_incorrect
        logging.info(
            f"matches: total:{total}  matched:{self.count_correct}  partial:{self.count_partial}  incorrect:{self.count_incorrect}"
        )

        # Persist into three separate files.
        self.__persist_as_jsonl(
            self.data_correct,
            filebase,
            DATA_CORRECT_STR + ".jsonl",
        )
        self.__persist_as_jsonl(
            self.data_partial,
            filebase,
            DATA_PARTLY_CORRECT_STR + ".jsonl",
        )
        self.__persist_as_jsonl(
            self.data_incorret,
            filebase,
            DATA_INCORRECT_STR + ".jsonl",
        )

        # reset everything
        model = self.model
        self.__init__()
        self.model = model

    def __run_batch(
        self,
        chunks: List[ChunkSubject],
        filebase: str = "",
        dry_run: bool = False,
    ) -> List[List[str]]:
        """
        Process a batch of chunks

        :param chunks: contains a list of chunks.
        :param filebase: the prefix of the file.
        :param dry_run: dont actually use the model, just create output files.
        """

        start = time.time()

        if dry_run:
            results = [["legal", "marketing"]] * len(chunks)
        else:
            inputs = []
            for c in chunks:
                inputs.append(c.text)
            results = self.model.classify(ClassificationRequest(input=inputs))

        end = time.time()
        logging.info(f"time taken: {end - start} seconds")

        # worth asserting
        assert len(results) == len(
            chunks
        ), f"results {len(results)}  chunks: {len(chunks)}"

        for index, chunk in enumerate(chunks):
            chunk.llm_labels = results[index]
            if isinstance(chunk.llm_labels, str):
                # Convert a single prompt to a list.
                chunk.llm_labels = [chunk.llm_labels]

            if chunk.label == chunk.llm_labels[0]:
                self.count_correct += 1
                self.data_correct.append(chunk)
            elif chunk.label in chunk.llm_labels:
                self.count_partial += 1
                self.data_partial.append(chunk)
            else:
                self.count_incorrect += 1
                self.data_incorret.append(chunk)

        self.__flush(filebase=filebase)

    def run(
        self,
        chunks: List[ChunkSubject],
        filebase: str,
        batch_size: int = -1,
        dry_run: bool = False,
    ) -> List[List[str]]:
        """
        Process chunks

        :param chunks: contains a list of chunks.
        :param filebase: the prefix of the file.
        :param batch_size: flush out files every batch.
        :param dry_run: dont actually use the model, just create output files.
        """
        if batch_size == -1 or batch_size > len(chunks):
            batch_size = len(chunks)

        index = 0
        while index < len(chunks):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(chunks))
            chunks_subset = chunks[start:end]
            if len(chunks_subset) == 0:
                break
            self.__run_batch(
                chunks=chunks_subset,
                filebase=f"{filebase}_{index}",
                dry_run=dry_run,
            )
            index += 1

    def model_name_to_model(self, model_name):
        if model_name == "deberta":
            from libdocs.classifiers.deberta.deberta import DebertaZeroShot

            return DebertaZeroShot()
        elif model_name == "mistral":
            from libdocs.classifiers.mistral.mistral import MistralInstruct

            return MistralInstruct()
        elif model_name == "zephyr":
            from libdocs.classifiers.zephyr.zephyr import ZephyrBeta

            return ZephyrBeta()
        else:
            raise ValueError(f"invalid model {model_name}")

    def model_run(
        self, model_name, chunks, file, intermediate_dir, output_dir, dry_run
    ):

        self.model = self.model_name_to_model(model_name)
        self.run(
            chunks,
            os.path.join(intermediate_dir, file),
            2000,
            dry_run,
        )

        # Combine the three sets of data
        for group in [
            DATA_CORRECT_STR,
            DATA_PARTLY_CORRECT_STR,
            DATA_INCORRECT_STR,
        ]:
            js = JSONL()
            js.from_files(
                intermediate_dir,
                "*_{group}_data.jsonl",
            )
            js.to_file(
                output_dir,
                f"{model_name}_{file}_{group}",
            )

    def add_model_verdict(
        self,
        model_name,
        input_dir,
        input_file,
        text_label,
        subject_label,
        intermediate_dir,
        dry_run=False,
    ):

        banner([f"Using model: {model_name}"])
        # Combine everything into combinded.jsonl
        js = JSONL()
        df = js.from_files(input_dir, input_file)

        # Create
        chunks = []
        texts = df[text_label]
        subjects = df[subject_label]
        for index, chunk in enumerate(texts):
            chunks.append(ChunkSubject(label=subjects[index], text=chunk))

        start = time.time()

        self.model_run(
            model_name,
            chunks,
            input_file.replace(".jsonl", ""),
            intermediate_dir,
            input_dir,
            dry_run,
        )

        end = time.time()
        print(f"Time taken: {end - start} seconds")
