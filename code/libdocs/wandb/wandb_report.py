import json
import logging
import os
from datetime import datetime

import pytz

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WandbMetricsReport:
    def __init__(self, wandb_run_path):
        """
        Initialize the WandbMetricsReport class with the path to Wandb run data.

        Parameters:
        - wandb_run_path: str, the path to the local directory containing Wandb run data.
        """
        self.wandb_run_path = wandb_run_path

    def read_metrics(self):
        """
        Read and parse the metrics from the wandb-summary.json file.

        Returns:
        - A dictionary containing the metrics data.
        """
        summary_file_path = os.path.join(
            self.wandb_run_path, "wandb-summary.json"
        )
        try:
            with open(summary_file_path, "r") as f:
                metrics = json.load(f)
        except FileNotFoundError:
            logging.info(f"Summary file not found in path: {summary_file_path}")
            return None
        return metrics

    def generate_html_content(self, metrics, others):
        """
        Generate HTML content from the parsed metrics, displaying each metric on a new row.

        Parameters:
        - metrics: dict, the metrics data extracted from Wandb summary.

        Returns:
        - A string containing the HTML representation of the metrics, formatted vertically.
        """

        # Define the Pacific Time Zone
        pacific_tz = pytz.timezone("America/Los_Angeles")

        # Get the current time in UTC and convert it to Pacific Time
        current_datetime_utc = datetime.now(pytz.utc)
        current_datetime_pacific = current_datetime_utc.astimezone(pacific_tz)

        # Format the date and time in Pacific Time
        formatted_datetime = current_datetime_pacific.strftime(
            "%I:%M %p on %B %d, %Y"
        )

        # Start building the HTML content
        html_content = (
            "<html><head><title>Fine tuning Report</title></head><body>"
        )
        html_content += f"<h1>Training Metrics as of {formatted_datetime}</h1>"
        html_content += "<table border='1'>"

        # Display additional information
        for key, value in others.items():
            html_content += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"

        # Add a row for each metric
        for key, value in metrics.items():
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html_content += "</table></body></html>"

        return html_content

    def save_html_report(self, html_content, output_path):
        """
        Save the generated HTML content to a file.

        Parameters:
        - html_content: str, the HTML content to be saved.
        - output_path: str, the path where the HTML file should be saved.
        """
        # Ensure the directory for the output path exists, create if necessary
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write the HTML content to the specified file
        with open(output_path, "w") as f:
            f.write(html_content)
        logging.info(f"Metrics report saved to {output_path}")

    def create_report(self, location, filename, others={}):
        """
        Main method to generate and save the metrics report as an HTML file.

        Parameters:
        - location: str, the path to the directory where the report should be saved.
        - filename: str, the name of the HTML file to be saved.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(location, exist_ok=True)
        output_path = os.path.join(location, filename)
        metrics = self.read_metrics()
        if metrics:
            html_content = self.generate_html_content(metrics, others)
            self.save_html_report(html_content, output_path)
        else:
            logging.info("No metrics found. Report not created.")


# Usage
if __name__ == "__main__":
    others = {"Train Dataset": 100}
    # Path to the wandb run directory
    wandb_run_path = "wandb/run-20240222_012922-gqme942n/files"
    location = "./outx"  # Output directory
    filename = "metrics2.html"  # Output filename
    report_generator = WandbMetricsReport(wandb_run_path)
    report_generator.create_report(location, filename, others=others)
