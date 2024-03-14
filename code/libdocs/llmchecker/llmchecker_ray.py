import json
import os

import pandas as pd
import ray
import ray.data
from libdocs.classifiers.common.types import ClassificationRequest

DATA_CORRECT_STR = "correct_data"
DATA_PARTLY_CORRECT_STR = "partly_correct_data"
DATA_INCORRECT_STR = "incorrect_data"


@ray.remote
class FileSink:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, "wb")

    def write_row(self, row: pd.Series):
        d = row.to_dict()
        # NOTE: this unfortunately flattens the model labels columns
        # pd.DataFrame(d).to_json(self.file, orient="records", lines=True)
        buf = (
            json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self.file.write(buf)
        self.file.flush()

    def write_batch(self, batch: pd.DataFrame):
        batch.to_json(self.file, orient="records", lines=True)
        self.file.flush()

    def __del__(self):
        if hasattr(self, "file") and self.file:
            self.file.flush()
            self.file.close()


class Classifier:
    def __init__(self, model_name: str, text_label: str, subject_label: str):
        self.model = self.model_name_to_model(model_name)
        self.model_name = model_name
        self.text_label: str = text_label
        self.subject_label: str = subject_label

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

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch[f"{self.model_name}_labels"] = self.model.classify(
            ClassificationRequest(input=batch[self.text_label])
        )
        return batch


class LLMChecker:
    def add_model_verdict(
        self,
        model_name: str,
        model_instances: int,
        input_file: str,
        text_label: str = "text",
        subject_label: str = "label",
        intermediate_dir: str = "/tmp/",
        dry_run: bool = False,
    ):
        # create filesink actors first
        inpath = os.path.abspath(input_file)
        outfilebase = (
            f"{os.path.splitext(os.path.basename(inpath))[0]}.{model_name}"
        )
        outfile_all = os.path.join(intermediate_dir, outfilebase + ".jsonl")
        outfile_correct = os.path.join(
            intermediate_dir, f"{outfilebase}.{DATA_CORRECT_STR}.jsonl"
        )
        outfile_partial = os.path.join(
            intermediate_dir, f"{outfilebase}.{DATA_PARTLY_CORRECT_STR}.jsonl"
        )
        outfile_incorrect = os.path.join(
            intermediate_dir, f"{outfilebase}.{DATA_INCORRECT_STR}.jsonl"
        )
        out_all = FileSink.options(name=model_name).remote(outfile_all)
        out_correct = FileSink.options(name=DATA_CORRECT_STR).remote(
            outfile_correct
        )
        out_partial = FileSink.options(name=DATA_PARTLY_CORRECT_STR).remote(
            outfile_partial
        )
        out_incorrect = FileSink.options(name=DATA_INCORRECT_STR).remote(
            outfile_incorrect
        )

        # run model
        ds = ray.data.read_json(f"local://{inpath}").map_batches(
            Classifier,
            fn_constructor_args=(model_name, text_label, subject_label),
            concurrency=model_instances,
            batch_size=32,
            num_gpus=0.01,
            batch_format="pandas",
        )

        # iterate over entries, and write them out into their correct categories
        count_incorrect = 0
        count_correct = 0
        count_partial = 0
        for batch in ds.iter_batches(
            batch_size=32, batch_format="pandas", drop_last=False
        ):
            res = []
            res.append(out_all.write_batch.remote(batch.copy()))
            for _, row in batch.iterrows():
                if row[subject_label] == row[f"{model_name}_labels"][0]:
                    count_correct += 1
                    res.append(out_correct.write_row.remote(row))
                elif row[subject_label] in row[f"{model_name}_labels"]:
                    count_partial += 1
                    res.append(out_partial.write_row.remote(row))
                else:
                    count_incorrect += 1
                    res.append(out_incorrect.write_row.remote(row))
            ray.get(res)

        print(f"{count_incorrect=} {count_correct=} {count_partial=}")
