import argparse
import logging
from datetime import datetime

from libdocs.google.bigquery import GoogleBigQueryProcessor


def main():
    """
    Takes the model verdicts file and loads it into a big query database of docspipeline-%Y%m%d
    """
    today_date = datetime.now()
    today_string = today_date.strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-gcs-uri",
        dest="uri",
        type=str,
        default="gs://docprocessor/all/model-verdicts.jsonl",
        help="The GCS input URI to load the data from.",
    )
    parser.add_argument(
        "--output-bq-dataset",
        dest="dataset",
        type=str,
        default="docspipeline",
        help="The Big Query dataset. Defaults to 'docspipeline'.",
    )
    parser.add_argument(
        "--output-bq-table",
        dest="table",
        type=str,
        default=today_string,
        help=f"The big query table. Defaults to '{today_string}'.",
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

    # create the big query processor and load it
    gbq = GoogleBigQueryProcessor("docspipeline", today_string)
    logging.info(
        f"Ensuring Big Query dataset '{args.dataset}' and table '{args.table}' exist and are created with the right schema..."
    )
    gbq.create()
    logging.info(
        f"Loading data into Big Query destination '{args.dataset}/{args.table}' from '{args.uri}'..."
    )
    gbq.load_from_gcs_uri(args.uri)
    logging.info("Done")


if __name__ == "__main__":
    main()
