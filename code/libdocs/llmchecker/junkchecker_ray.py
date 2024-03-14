import json
import logging
import os

import pandas as pd
import ray
import ray.data
from libdocs.classifiers.common.prompt import junk_classifier_prompt
from libdocs.classifiers.common.types import ClassificationRequest
from numpy import ndarray

DATA_JUNK_STR = "junk_data"
DATA_CLEAN_STR = "clean_data"
DATA_UNKNOWN_STR = "unknown_data"


@ray.remote
class FileSink:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, "wb")

    def write_row(self, row: pd.Series):
        try:
            d = row.to_dict()
            # This is pretty annoying: even though we get pandas batches, lists might be numpy arrays
            # this affects all of our labels which have been previously lists
            for k, v in d.items():
                if isinstance(v, ndarray):
                    d[k] = v.tolist()
            # NOTE: this unfortunately flattens the model labels columns
            # pd.DataFrame(d).to_json(self.file, orient="records", lines=True)
            buf = (
                json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n"
            ).encode("utf-8")
            self.file.write(buf)
            self.file.flush()
        except Exception as err:
            logging.error(
                f"failed to write row to file '{self.filename}': {err}: {d=} (type = {type(d)})"
            )
            # for k, v in d.items():
            #     logging.error(f"{k}: {type(v)}")

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
        if model_name == "mistral":
            from libdocs.classifiers.mistral.mistral import MistralInstruct

            return MistralInstruct()
        # NOTE: zephyr doesn't work yet - we should investigate why
        # elif model_name == "zephyr":
        #     from libdocs.classifiers.zephyr.zephyr import ZephyrBeta
        #
        #    return ZephyrBeta()
        else:
            raise ValueError(f"invalid model {model_name}")

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch[f"{self.model_name}_junk_labels"] = self.model.classify(
            ClassificationRequest(input=batch[self.text_label]),
            prompt=junk_classifier_prompt,
        )
        return batch


class JunkChecker:
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
            f"{os.path.splitext(os.path.basename(inpath))[0]}.junk_{model_name}"
        )
        outfile_all = os.path.join(intermediate_dir, outfilebase + ".jsonl")
        outfile_clean = os.path.join(
            intermediate_dir, f"{outfilebase}.{DATA_CLEAN_STR}.jsonl"
        )
        outfile_unknown = os.path.join(
            intermediate_dir, f"{outfilebase}.{DATA_UNKNOWN_STR}.jsonl"
        )
        outfile_junk = os.path.join(
            intermediate_dir, f"{outfilebase}.{DATA_JUNK_STR}.jsonl"
        )
        out_all = FileSink.options(name=model_name).remote(outfile_all)
        out_clean = FileSink.options(name=DATA_CLEAN_STR).remote(outfile_clean)
        out_unknown = FileSink.options(name=DATA_UNKNOWN_STR).remote(
            outfile_unknown
        )
        out_junk = FileSink.options(name=DATA_JUNK_STR).remote(outfile_junk)

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
        count_junk = 0
        count_clean = 0
        count_unknown = 0
        for batch in ds.iter_batches(
            batch_size=32, batch_format="pandas", drop_last=False
        ):
            res = []
            res.append(out_all.write_batch.remote(batch.copy()))
            for _, row in batch.iterrows():
                if row[f"{model_name}_junk_labels"][0] == "clean":
                    count_clean += 1
                    res.append(out_clean.write_row.remote(row))
                elif row[f"{model_name}_junk_labels"][0] == "junk":
                    count_junk += 1
                    res.append(out_junk.write_row.remote(row))
                else:
                    count_unknown += 1
                    res.append(out_unknown.write_row.remote(row))
            ray.get(res)

        logging.info(f"{count_junk=} {count_clean=} {count_unknown=}")
