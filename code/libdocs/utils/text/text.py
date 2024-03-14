import logging
import re
from typing import Callable, Iterable, List, Optional, TypeAlias

from pandas import DataFrame, Series
from pydantic import BaseModel, Field, field_validator


class Row(BaseModel):
    """
    Class to hold a row of input data
    """

    text: str = Field(..., description="The text chunk")
    label: str = Field(..., description="The input label / subject / topic")
    input_src: str = Field(..., description="The input source of the text")
    entity_id: str = Field(
        ..., description="The stable entity ID of the text (SHA-256 hexstring)"
    )
    id: Optional[int] = Field(None, description="The id of the text chunk.")
    deberta_labels: Optional[List[str]] = Field(
        None, description="The deberta classification labels"
    )
    mistral_labels: Optional[List[str]] = Field(
        None, description="The mistral classification labels"
    )
    zephyr_labels: Optional[List[str]] = Field(
        None, description="The zephyr classification labels"
    )
    deberta_verdict: Optional[str] = Field(
        None, description="The deberta classification verdict"
    )
    mistral_verdict: Optional[str] = Field(
        None, description="The mistral classification verdict"
    )
    zephyr_verdict: Optional[str] = Field(
        None, description="The zephyr classification verdict"
    )

    @field_validator("text", "label", "input_src", "entity_id")
    @classmethod
    def text_length(cls, v):
        if len(v) == 0:
            raise ValueError("must be non-empty")
        return v

    @field_validator("deberta_labels", "mistral_labels", "zephyr_labels")
    @classmethod
    def labels_length(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("if set, the list must not be empty")
        return v

    @classmethod
    def from_series(cls, row: Series):
        """
        Creates a Row from a pandas Series
        """
        return cls.model_validate(row.to_dict())

    def to_series(self, base_series: Series = Series()):
        """
        Converts a row back to a pandas Series. Optionally, it takes a base series
        which will be copied and used as the base for the return.
        """
        ret = base_series.copy()

        # required fields we simply assign back
        ret["text"] = self.text
        ret["label"] = self.label
        ret["input_src"] = self.input_src
        ret["entity_id"] = self.entity_id

        # optionals are a bit more involving
        if self.id is None:
            if "id" in ret.index:
                ret.drop(index="id", inplace=True)
        else:
            ret["id"] = self.id

        if self.deberta_labels is None:
            if "deberta_labels" in ret.index:
                ret.drop(index="deberta_labels", inplace=True)
        else:
            ret["deberta_labels"] = self.deberta_labels

        if self.mistral_labels is None:
            if "mistral_labels" in ret.index:
                ret.drop(index="mistral_labels", inplace=True)
        else:
            ret["mistral_labels"] = self.mistral_labels

        if self.zephyr_labels is None:
            if "zephyr_labels" in ret.index:
                ret.drop(index="zephyr_labels", inplace=True)
        else:
            ret["zephyr_labels"] = self.zephyr_labels

        if self.deberta_verdict is None:
            if "deberta_verdict" in ret.index:
                ret.drop(index="deberta_verdict", inplace=True)
        else:
            ret["deberta_verdict"] = self.deberta_verdict

        if self.mistral_verdict is None:
            if "mistral_verdict" in ret.index:
                ret.drop(index="mistral_verdict", inplace=True)
        else:
            ret["mistral_verdict"] = self.mistral_verdict

        if self.zephyr_verdict is None:
            if "zephyr_verdict" in ret.index:
                ret.drop(index="zephyr_verdict", inplace=True)
        else:
            ret["zephyr_verdict"] = self.zephyr_verdict

        return ret


MutatingRowFunc: TypeAlias = Callable[[Row], Row]
DropRowFunc: TypeAlias = Callable[[Row], bool]


def __apply_drop_func(f: DropRowFunc):
    def apply(series: Series) -> bool:
        try:
            row = Row.from_series(series)
            return f(row)
        except Exception as err:
            logging.error(
                f"{series.index}: failed to convert to Row: {err}. Dropping row."
            )
            return True

    return apply


def __apply_mutate_func(f: MutatingRowFunc):
    def apply(series: Series) -> Series:
        try:
            row = Row.from_series(series)
            mutated_row = f(row)
            return mutated_row.to_series(series)
        except Exception as err:
            logging.error(
                f"{series.index}: failed to convert to Row: {err}. Returning row as-is."
            )
            return series

    return apply


# DROP FUNCTIONS


def not_all_models_agree(row: Row) -> bool:
    """
    Filters out rows where the models don't agree with each other
    """
    return not (
        row.deberta_labels[0] == row.mistral_labels[0]
        and row.deberta_labels[0] == row.zephyr_labels[0]
    )


def not_model_majority(row: Row) -> bool:
    """
    Keeps only the rows where a majority of the models agree
    """
    return not (
        row.deberta_labels[0] == row.mistral_labels[0]
        or row.deberta_labels[0] == row.zephyr_labels[0]
        or row.mistral_labels[0] == row.zephyr_labels[0]
    )


__valid_labels: List[str] = [
    "financial",
    "strategy_and_planning",
    "sales",
    "marketing",
    "technical",
    "legal",
    "risk_and_compliance",
    "human_resource",
    "cybersecurity",
    "business_development",
    # NOTE: commented out because the topic is too close to the others
    # and subjects should be as distinct as possible for better results
    # "business_ethics",
]


def not_a_valid_label(row: Row) -> bool:
    """
    Throws out rows that have a final label that is not within the list of valid labels
    """
    return row.label not in __valid_labels


def any_conversation(row: Row) -> bool:
    """
    Throws out rows where any of the models came to the conclusion that this is a conversation.
    """
    return (
        (
            row.deberta_labels is not None
            and len(row.deberta_labels) > 0
            and row.deberta_labels[0] == "conversation"
        )
        or (
            row.mistral_labels is not None
            and len(row.mistral_labels) > 0
            and row.mistral_labels[0] == "conversation"
        )
        or (
            row.zephyr_labels is not None
            and len(row.zephyr_labels) > 0
            and row.zephyr_labels[0] == "conversation"
        )
    )


def any_irrelevant(row: Row) -> bool:
    """
    Throws out rows where any of the models came to the conclusion that the rows are irrelevant.
    """
    return (
        (
            row.deberta_labels is not None
            and len(row.deberta_labels) > 0
            and row.deberta_labels[0] == "irrelevant"
        )
        or (
            row.mistral_labels is not None
            and len(row.mistral_labels) > 0
            and row.mistral_labels[0] == "irrelevant"
        )
        or (
            row.zephyr_labels is not None
            and len(row.zephyr_labels) > 0
            and row.zephyr_labels[0] == "irrelevant"
        )
    )


def short_sentences(row: Row) -> bool:
    """
    Throw out rows that have less than 15 words (average English sentence length is 15-20 words).
    """
    return len(row.text.split()) < 15


def contains_url(row: Row) -> bool:
    """
    Removes rows that contains a URL
    """
    return "http" in row.text or "HTTP" in row.text or "Http" in row.text


def text_starts_with_alpha(row: Row) -> bool:
    """
    Removes rows where the text does not start with an alpha character
    """
    return not row.text[0].isalpha()


def contains_chapter_reference(row: Row) -> bool:
    """
    Removes rows that have 'chapter' references because these are most likely
    coming from titles in books
    """
    return "chapter" in row.text.lower()


__year_re = re.compile(r".*(\([12][0-9]{3}\)|\[[12][0-9]{3}\]).*")


def contains_year_reference(row: Row) -> bool:
    """
    Removes rows that have years in parantheses or brackets because these are most likely
    references to books (like 'Davidson (2017)')
    """
    return True if __year_re.match(row.text) is not None else False


def is_diagram_description(row: Row) -> bool:
    """
    Removes rows which are most likely descriptions of diagrams that start with: 'Figure 1.6 ...'.
    """
    return row.text.lower().startswith("figure ")


def is_table_description(row: Row) -> bool:
    """
    Removes rows which are most likely descriptions of tables that start with: 'Table 3.2 ...'.
    """
    return row.text.lower().startswith("table ")


ALL_DROP_ROW_FUNCS: List[DropRowFunc] = [
    not_all_models_agree,
    not_model_majority,
    not_a_valid_label,
    any_conversation,
    any_irrelevant,
    short_sentences,
    text_starts_with_alpha,
    contains_url,
    contains_chapter_reference,
    contains_year_reference,
    is_diagram_description,
    is_table_description,
]

DEFAULT_DROP_ROW_FUNCS: List[DropRowFunc] = [
    not_model_majority,
    not_a_valid_label,
    any_conversation,
    any_irrelevant,
    text_starts_with_alpha,
    contains_url,
    contains_chapter_reference,
    contains_year_reference,
    is_diagram_description,
    is_table_description,
]


# MUTATING FUNCTIONS


def all_models_disagreeing_with_input(row: Row) -> Row:
    """
    Changes 'label' to the label of all models as they all came to the same conclusion,
    and our input is most likely wrong. I also set all model verdict columns to 'input_incorrect'.
    Precondition is that the current label is not 'unlabeled'.
    """
    if (
        row.label != "unlabeled"
        and row.deberta_labels[0] == row.mistral_labels[0]
        and row.deberta_labels[0] == row.zephyr_labels[0]
        and row.deberta_labels[0] != row.label
    ):
        row.label = row.deberta_labels[0]
        row.deberta_verdict = "input_incorrect"
        row.mistral_verdict = "input_incorrect"
        row.zephyr_verdict = "input_incorrect"

    return row


def ensure_only_ascii_characters(row: Row) -> Row:
    """
    Throws out rows where the text has non-ASCII characters
    """
    if not row.text.isascii():
        row.text = row.text.encode("ascii", "ignore").decode().strip()
    return row


def reverse_labeling(row: Row) -> Row:
    """
    Apply reverse labeling: for all 'unlabeled' rows, it will take the best label.
    """
    if row.label == "unlabeled":
        # all labels match
        if (
            row.deberta_labels[0] == row.mistral_labels[0]
            and row.deberta_labels[0] == row.zephyr_labels[0]
        ):
            row.label = row.deberta_labels[0]
        # mistral + zephhyr match
        elif row.mistral_labels[0] == row.zephyr_labels[0]:
            row.label = row.mistral_labels[0]
        # mistral + deberta match
        elif row.mistral_labels[0] == row.deberta_labels[0]:
            row.label = row.mistral_labels[0]
        # zephyr + deberta match
        elif row.zephyr_labels[0] == row.deberta_labels[0]:
            row.label = row.zephyr_labels[0]
        # we take the mistral label if there are no majorities
        else:
            row.label = row.mistral_labels[0]
        # now set the verdicts to 'n/a' because we did reverse labeling
        row.deberta_verdict = "n/a"
        row.mistral_verdict = "n/a"
        row.zephyr_verdict = "n/a"
    return row


ALL_MUTATING_ROW_FUNCS: List[MutatingRowFunc] = [
    reverse_labeling,
    all_models_disagreeing_with_input,
    ensure_only_ascii_characters,
]

DEFAULT_MUTATING_ROW_FUNCS: List[MutatingRowFunc] = [
    reverse_labeling,
    all_models_disagreeing_with_input,
    ensure_only_ascii_characters,
]


def dedup_input_data(df: DataFrame):
    """
    This deduplicates the input data based on the 'entity_id' column making sure to keep one entry though.
    NOTE: This mutates the passed DataFrame. Pass a copy if you need to preserve the input DataFrame
    """
    df.drop_duplicates(subset=["entity_id"], keep="first", inplace=True)


def drop_input_data(
    in_df: DataFrame, funcs: Iterable[DropRowFunc] = DEFAULT_DROP_ROW_FUNCS
) -> DataFrame:
    """
    This cleans up input data with a list of cleanup/dropping functions, and returns the cleaned up DataFrame.
    """
    total_drops = 0
    df = in_df
    for f in funcs:
        logging.debug(f"applying cleanup function: '{f.__name__}()'...")
        to_drop = df.apply(__apply_drop_func(f), axis=1)
        to_drop_count = to_drop.sum()
        total_drops += to_drop_count
        logging.info(
            f"cleanup function '{f.__name__}()' is dropping {to_drop_count} rows"
        )
        df = df[~to_drop]
    logging.info(f"cleanup_input_data() dropped {total_drops} rows in total")
    return df


def mutate_input_data(
    in_df: DataFrame,
    funcs: Iterable[MutatingRowFunc] = DEFAULT_MUTATING_ROW_FUNCS,
) -> DataFrame:
    """
    This applies a set of data mutating functions, and returns with the mutated DataFrame.
    """
    df = in_df
    for f in funcs:
        logging.debug(f"applying mutating function: '{f.__name__}()'...")
        df = df.apply(__apply_mutate_func(f), axis=1)
        logging.info(f"applied mutating function '{f.__name__}()'")
    return df
