import os

import libdocs.utils.label.label as label_util
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from libdocs.utils.jsonl.jsonl import JSONL
from plotly.subplots import make_subplots


# Plot verdict for a model
def plot_verdict(
    df: pd.DataFrame,
    figure: go.Figure,
    rowid: int,
    model: str,
    verdict: str,
    subject: str,
    id: str,
):
    df = df.groupby([verdict, subject], as_index=False).count()
    bar = px.bar(
        df,
        x=df[subject],
        y=df[id],
        color=verdict,
        barmode="group",
        title=f"{model} verdict",
    )
    figure.add_traces(data=bar["data"], rows=rowid, cols=1)
    figure.update_xaxes(title=f"{model} subject verdicts", row=rowid, col=1)
    figure.update_yaxes(title="counts", row=rowid, col=1)


# Plot verdict for all models
def create_model_verdicts_plot(
    df: pd.DataFrame,
    file: str = None,
    save: bool = False,
    show: bool = False,
    models: list[str] = ["deberta", "mistral", "zephyr"],
):

    figure = make_subplots(rows=len(models) + 1, cols=1)

    for index, model in enumerate(models):
        verdict = model + "_verdict"
        plot_verdict(
            df.copy(deep=True),
            figure,
            index + 1,
            model,
            verdict,
            "label",
            "entity_id",
        )

    figure.update_layout(title="Subject Frequency Table")

    # Breakdown Chart
    newdf = df.groupby(
        [
            "label",
            "zephyr_verdict",
            "mistral_verdict",
            "deberta_verdict",
        ],
        as_index=False,
    ).count()
    bar = px.bar(
        newdf,
        x=newdf["label"],
        y=newdf["entity_id"],
        color=verdict,
        barmode="group",
        title="combined verdict",
        hover_data=[
            "label",
            # "zephyr_verdict",
            # "mistral_verdict",
            # "deberta_verdict",
            # "entity_id",
        ],
    )
    figure.add_traces(data=bar["data"], rows=4, cols=1)
    figure.update_xaxes(title="subjects", row=4, col=1)
    figure.update_yaxes(title="counts", row=4, col=1)
    figure.update_legends()

    if save:
        figure.write_html(file)
        figure.write_image(file.replace(".html", ".png"))
    if show:
        figure.show()


def discover_labels(df: pd.DataFrame) -> list[str]:

    models: list[str] = ["deberta", "mistral", "zephyr"]

    subjects = df["label"].unique().tolist()
    subjects.append("conversation")
    subjects.append("irrelevant")
    subjects.append("not_safe_for_workplace")

    discovered = []
    for model in models:

        # check if labels can be fixed
        for labels in df[model + "_labels"]:
            _, discovered_labels = label_util.sanitize(subjects, labels)
            discovered += discovered_labels

    return discovered


if __name__ == "__main__":

    import argparse

    """Visualization."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--input-dir",
        default="data/feb22",
        help="input directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/feb22",
        help="output directory",
    )
    parser.add_argument(
        "-f",
        "--input-file",
        default="model-verdicts.jsonl",
        help="input jsonl file. wildcards also work.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="save html.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show html.",
    )
    args = parser.parse_args()

    j = JSONL()
    j.from_files(args.input_dir, args.input_file)
    discovered_labels = discover_labels(j.df)

    # Create Graphs
    create_model_verdicts_plot(
        j.df,
        os.path.join(
            args.input_dir, args.input_file.replace(".jsonl", ".html")
        ),
        args.save,
        args.show,
    )

    # Store discovered labels
    df = pd.DataFrame(columns=["discovered_labels"])
    df["discovered_labels"] = list(set(discovered_labels))
    discovered = JSONL(df)
    discovered.to_file(
        args.output_dir, args.input_file.replace(".jsonl", ".discovered")
    )
