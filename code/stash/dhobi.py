import csv
import glob
import logging as logger
import os
import time
from threading import Thread

import torch
from libdocs.classifiers.acuopenai.acuopenai import AcuOpenAI
from libdocs.classifiers.common.types import ClassificationRequest
from libdocs.classifiers.deberta.deberta import DebertaZeroShot
from libdocs.classifiers.mistral.mistral import MistralInstruct
from libdocs.classifiers.zephyr.zephyr import ZephyrBeta
from libdocs.types.types import ChunkSubject
from rich import print as rprint


class Dhobi:
    # ---------------------------------------------------------------------------------------------
    def __init__(
        self, csv_file: str, output_dir: str, model_name: str, model=None
    ) -> None:
        if not csv_file:
            raise ValueError("csv_file must be non-empty")

        # check_valid_file(csv_file)
        self.csv_file = csv_file
        self.output_file_base = os.path.join(
            output_dir, os.path.basename(csv_file)
        )
        self.output_file_base = self.output_file_base.replace(
            ".csv", "_" + model_name + "_"
        )

        if model is None:
            if model_name == "zephyr":
                model = ZephyrBeta()
            elif model_name == "mistral":
                model = MistralInstruct()
            elif model_name == "deberta":
                model = DebertaZeroShot()
            else:
                model = AcuOpenAI()
        self.model = model

    # ---------------------------------------------------------------------------------------------
    def load_csv(self) -> [ChunkSubject]:
        """
        Load the csv file containing the columns: label, text

        :return: a list of ChunkSubject objects
        """
        logger.info(f"Loading the csv file: {self.csv_file}")
        chunks = []
        with open(self.csv_file, "r") as file:
            reader = csv.DictReader(file)
            chunks = [ChunkSubject(**row) for row in reader]
        logger.info(f"Loaded {len(chunks)} chunks from the csv file.")
        return chunks

    # ---------------------------------------------------------------------------------------------

    def persist_as_jsonl(self, chunks: [ChunkSubject], file_name: str) -> None:
        """
        Persist the chunk to the file as jsonl

        :param chunks: the chunks to be persisted.
        :return: None
        """
        # pre-conditions check
        if not chunks:
            raise ValueError("chunks must be non-empty")
        if not file_name:
            raise ValueError("file_name must be non-empty")

        logger.info(
            f"Persisting the {len(chunks)}-chunks to the file: {file_name}"
        )
        with open(file_name, "w") as file:
            for chunk in chunks:
                file.write(chunk.json() + "\n")
        logger.info(
            f"Persisted the {len(chunks)}-chunks to the file: {file_name}"
        )

    # ---------------------------------------------------------------------------------------------

    def run(self, dry_run):
        subset = self.load_csv()
        total_count = len(subset)
        current_count = 0
        with_correct = 0
        incorrect_count = 0

        consensus_data = []
        partly_correct_data = []
        incorrect_data = []

        print("=" * 120)

        start = time.time()

        multiple_chunks = []
        results = []
        for index, chunk in enumerate(subset):
            multiple_chunks.append(chunk.text)
            results.append(["legal", "marketing"])

        if not dry_run:
            start = time.time()
            results = self.model.classify(
                ClassificationRequest(input=multiple_chunks)
            )

        # worth asserting
        rprint(f"results {len(results)}  subset: {len(subset)}")
        assert len(results) == len(subset)

        for index, chunk in enumerate(subset):
            chunk.llm_labels = results[index]
            if isinstance(chunk.llm_labels, str):
                # Convert a single prompt to a list.
                chunk.llm_labels = [chunk.llm_labels]

            rprint(chunk)

            if chunk.label == chunk.llm_labels[0]:
                current_count += 1
                consensus_data.append(chunk)
            elif chunk.label in chunk.llm_labels:
                with_correct += 1
                partly_correct_data.append(chunk)
            else:
                incorrect_count += 1
                incorrect_data.append(chunk)

        rprint(
            f"Match counts: total:{total_count}  matched:{current_count}  partial:{with_correct}  incorrect:{incorrect_count}"
        )

        end = time.time()
        rprint(f"Time taken: {end - start} seconds")

        # Persist into three separate files.
        if len(consensus_data) != 0:
            self.persist_as_jsonl(
                consensus_data,
                self.output_file_base + "consensus_data.jsonl",
            )
        if len(partly_correct_data) != 0:
            self.persist_as_jsonl(
                partly_correct_data,
                self.output_file_base + "partly_correct_data.jsonl",
            )
        if len(incorrect_data) != 0:
            self.persist_as_jsonl(
                incorrect_data,
                self.output_file_base + "incorrect_data.jsonl",
            )


def single_dhobi_run(
    dhobifile: str,
    outputdirectory: str,
    dry_run: bool,
    model_name: str,
    model=None,
):
    dhobi = Dhobi(
        dhobifile, outputdirectory, model_name=model_name, model=model
    )
    dhobi.run(dry_run)


def flush_gpu_memory_allocations():
    import gc

    import torch
    from accelerate.utils import release_memory

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    release_memory()


if __name__ == "__main__":
    print("start of main")
    print(torch.cuda.memory_summary())

    dir_path = os.path.dirname(os.path.realpath(__file__))
    BASEDIR = os.path.join(dir_path, "../data")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-directory",
        type=str,
        default=f"{BASEDIR}",
        help="base directory.",
    )
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        default="splitcsv",
        help="input directory.",
    )
    parser.add_argument(
        "-s",
        "--set",
        type=str,
        default="set1",
        help="set.",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="dry run.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        default="splitcsvoutput",
        help="output directory.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deberta",
        help="model - zephyr|mistral|openai|deberta.",
    )
    parser.add_argument(
        "-n", "--num-threads", type=int, default=0, help="num threads."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose.",
    )
    args = parser.parse_args()

    input_dir = os.path.join(args.base_directory, args.input_directory)
    input_dir = os.path.join(input_dir, args.set)
    files = glob.glob(input_dir + "/*", recursive=True)
    output_dir = os.path.join(args.base_directory, args.output_directory)
    output_dir = os.path.join(output_dir, args.set)

    flush_gpu_memory_allocations()

    if args.model == "zephyr":
        model = ZephyrBeta()
    elif args.model == "mistral":
        model = MistralInstruct()
    elif args.model == "deberta":
        if args.verbose:
            print("before deberta")
            print(torch.cuda.memory_summary())
        model = DebertaZeroShot()
        if args.verbose:
            print("after deberta")
            print(torch.cuda.memory_summary())
    else:
        model = AcuOpenAI()

    # Fire up multiple instances of threads.
    threads = []

    if args.verbose:
        print("after model initialization")
        print(torch.cuda.memory_summary())

    parallel = False
    for index, file in enumerate(files):
        if not parallel:

            if args.verbose:
                print(f"begin loop iteration {index}")
                print(torch.cuda.memory_summary())

            single_dhobi_run(
                dhobifile=file,
                outputdirectory=output_dir,
                dry_run=args.dry_run,
                model_name=args.model,
                model=model,
            )

            # Be nice release all memory
            # flush_gpu_memory_allocations()

            if args.verbose:
                print(f"end loop iteration {index}")
                print(torch.cuda.memory_summary())

        else:

            def task():
                single_dhobi_run(
                    dhobifile=file,
                    outputdirectory=output_dir,
                    dry_run=args.dry_run,
                    model_name=args.model,
                    model=model,
                )

            thread = Thread(target=task)
            threads.append(thread)
            thread.start()

    # Wait for threads to finish
    for thread in threads:
        print("Waiting for the thread to finish...")
        thread.join()

    # Be nice release all memory
    flush_gpu_memory_allocations()
