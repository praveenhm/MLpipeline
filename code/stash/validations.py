import argparse
import logging

from libdocs.google.bigquery import GoogleBigQueryProcessor
from libdocs.utils.jsonl.jsonl import JSONL

GCP_BQ_DATASET = "training"
GCP_BQ_TABLE = "pdfs"


def main():
    """Runs all the steps of the docspipeline from splitting to the umap analyzer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcp-bq-dataset",
        dest="gcpBqDataset",
        type=str,
        default=GCP_BQ_DATASET,
        help="the Big Query dataset to use (step 7)",
    )
    parser.add_argument(
        "--gcp-bq-table",
        dest="gcpBqTable",
        type=str,
        default=GCP_BQ_TABLE,
        help="the Big Query table to use (step 7)",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    gbq = GoogleBigQueryProcessor(args.gcpBqDataset, args.gcpBqTable)
    o = gbq.lookup(QUERY="SELECT * FROM " + gbq.table_id)  # + " LIMIT 100")
    p = o.to_dataframe()
    j = JSONL(p)
    j.to_file("./", "bq")


if __name__ == "__main__":
    main()
