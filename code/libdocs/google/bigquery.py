import hashlib
import logging

import google.api_core.exceptions as exceptions
from google.cloud import bigquery


class GoogleBigQueryProcessor:
    """
    GoogleBigQueryProcessor is capable of interacting with BigQuery.
    """

    def __init__(self, dataset="training", table="pdfs"):
        """
        Construct a new 'GoogleBigQueryProcessor' object.

        :param project: The project name for GCS.
        :param dataset: The dataset name.
        :return: returns nothing
        """
        # Construct a BigQuery client object.
        self.client = bigquery.Client()
        self.dataset_name = dataset
        self.dataset_id = "{}.{}".format(self.client.project, self.dataset_name)
        self.dataset = None
        self.datasets = list(
            self.client.list_datasets(self.client.project, timeout=30)
        )
        for d in self.datasets:
            if d.dataset_id == dataset:
                self.dataset = self.client.get_dataset(
                    self.dataset_id, timeout=30
                )
                break
        self.table_name = table
        self.table_id = "{}.{}.{}".format(self.client.project, dataset, table)
        self.table = None
        if self.dataset is not None:
            for t in self.client.list_tables(self.dataset, timeout=30):
                if t.table_id == self.table_name:
                    self.table = self.client.get_table(
                        self.table_id, timeout=30
                    )
                    break

    def info(self) -> str:
        if self.dataset is None:
            return ""

        tables = list(self.client.list_tables(self.dataset))
        tables_str = ""
        if tables:
            for table in tables:
                tables_str += "       name: {}\n".format(table.table_id)
        else:
            tables_str = "       dataset has no tables defined."

        labels = self.dataset.labels
        label_str = ""
        if labels:
            for label, value in labels.items():
                label_str += "\t{}: {}\n".format(label, value)
        else:
            label_str = "dataset has no labels defined."

        out = "Dataset:\n"
        out += "       name: {}\n".format(self.dataset.dataset_id)
        out += "     labels: {}\n".format(label_str)
        out += "Tables:\n"
        out += tables_str
        return out

    def create(self):
        if self.dataset is None:
            dataset = bigquery.Dataset(self.dataset_id)
            dataset.location = "US"
            # Send the dataset to the API for creation, with an explicit timeout.
            # Raises google.api_core.exceptions.Conflict if the Dataset already
            # exists within the project.
            try:
                self.dataset = self.client.create_dataset(dataset, timeout=30)
                logging.info(f"Created dataset '{self.dataset_id}'")
            except exceptions.Conflict as err:
                # this cannot really happen because we are already testing above, but it doesn't hurt either to catch this
                logging.info(
                    f"Not creating dataset '{self.dataset_id}': dataset already exists (create error: {err})"
                )
        else:
            logging.info(
                f"Not creating dataset '{self.dataset_id}': dataset already exists"
            )

        if self.table is None:
            # create table
            # NOTE: all required fields are marked like this because we populate them at creation time
            schema = [
                bigquery.SchemaField("id", "INT64", mode="NULLABLE"),
                bigquery.SchemaField("entity_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("input_src", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("label", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("deberta_labels", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("mistral_labels", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("zephyr_labels", "JSON", mode="NULLABLE"),
                bigquery.SchemaField(
                    "deberta_verdict", "INT64", mode="NULLABLE"
                ),
                bigquery.SchemaField(
                    "mistral_verdict", "INT64", mode="NULLABLE"
                ),
                bigquery.SchemaField(
                    "zephyr_verdict", "INT64", mode="NULLABLE"
                ),
            ]
            table = bigquery.Table(self.table_id, schema=schema)
            self.table = self.client.create_table(table)
            logging.info(f"Created table '{self.table_id}'")
        else:
            logging.info(
                f"Not creating table '{self.table_id}': table already exists"
            )

    def lookup(self, QUERY=None):
        if QUERY is None:
            QUERY = "SELECT * FROM " + self.table_id
        query_job = self.client.query(QUERY)
        return query_job.result()

    def query(self, query):
        query_job = self.client.query(query)
        return query_job.result()

    def delete_for_gcs_uri(self, gcs_uri: str):
        # TODO: should get scalar paramters
        query_job = self.client.query(
            f"DELETE FROM {self.table_id} WHERE input_src = '{gcs_uri}'"
        )
        return query_job.result()

    def load_from_gcs_uri(self, gcs_uri: str):
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("id", "INT64", mode="NULLABLE"),
                bigquery.SchemaField("entity_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("input_src", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("label", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("deberta_labels", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("mistral_labels", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("zephyr_labels", "JSON", mode="NULLABLE"),
                bigquery.SchemaField(
                    "deberta_verdict", "STRING", mode="NULLABLE"
                ),
                bigquery.SchemaField(
                    "mistral_verdict", "STRING", mode="NULLABLE"
                ),
                bigquery.SchemaField(
                    "zephyr_verdict", "STRING", mode="NULLABLE"
                ),
            ],
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )

        load_job = self.client.load_table_from_uri(
            gcs_uri,
            self.table_id,
            location=self.dataset.location,  # Must match the destination dataset location.
            job_config=job_config,
        )
        return load_job.result()

    def insert(self, chunks: list[str], gcs_uri: str, topic: str):
        if self.table is None:
            raise Exception(
                "table does not exist: you must create the table first"
            )
        # iterate over all chunks and create a hash for it for the entity_id
        rows = []
        for chunk in chunks:
            entity_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "entity_id": entity_id,
                    "text": chunk,
                    "input_src": gcs_uri,
                    "label": topic,
                }
            )
        return self.client.insert_rows(self.table, rows)

    def update(self, values, matches: list[str], QUERY=None):
        vstr = ", ".join(values)
        mstr = ", ".join(matches)
        if QUERY is None:
            QUERY = "UPDATE {} SET {} WHERE {}".format(
                self.table_id, vstr, mstr
            )
        query_job = self.client.query(QUERY)
        return query_job.result()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    b = GoogleBigQueryProcessor()
    b.create()
    print(b.info())
