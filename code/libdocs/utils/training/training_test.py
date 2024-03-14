import pandas as pd
import pytest
from libdocs.utils.training.training import normalize_data


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests."""
    data = {
        "text": [
            "Hornegger: I like your statement that what you...",
            "Forty, fifty thousand students and staff together...",
            "2/10/24, 9:32 PM 6 Must-have Business Development...",
            "Business development frameworks are essential...",
            "Instead, you have to build smart processes that...",
        ],
        "hypothesis": [
            "Hornegger: I like your statement that what you...",
            "Forty, fifty thousand students and staff together...",
            "2/10/24, 9:32 PM 6 Must-have Business Development...",
            "Business development frameworks are essential...",
            "Instead, you have to build smart processes that...",
        ],
        "labels": [1, 1, 1, 1, 1],
        "task_name": [
            "business_development",
            "business_development",
            "business_development",
            "business_development",
            "business_development",
        ],
        "label_text": [
            "business_development",
            "business_development",
            "business_development",
            "business_development",
            "business_development",
        ],
    }
    return pd.DataFrame(data)


def test_normalize_data(sample_data):
    filter_labels = ["exclude_label"]
    normalized_df = normalize_data(sample_data, filter_labels, "label_text")

    # Check if filter_labels are excluded
    assert not set(filter_labels).intersection(
        normalized_df["label_text"].unique()
    ), "Filtered labels are not excluded properly."

    # Check if distribution is normalized within limits
    value_counts = normalized_df["label_text"].value_counts()
    min_count, max_count = value_counts.min(), value_counts.max()

    assert (
        min_count <= max_count <= 2 * min_count
    ), "Distribution is not normalized within the specified limits."

    # Check if the distribution is normalized within limits
    for label in normalized_df["label_text"].unique():
        assert (
            0
            < normalized_df[normalized_df["label_text"] == label].shape[0]
            <= 2 * min_count
        ), f"Distribution for {label} is not within specified limits."
