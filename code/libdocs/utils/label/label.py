import itertools
import re
from typing import Tuple

control_chars = "".join(
    map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0)))
)
control_char_re = re.compile("[%s]" % re.escape(control_chars))


def __clean_label(label: str) -> list[str]:
    label = re.sub(r"-", "_", label)
    label = re.sub(r"_+", "_", label)
    # remove all chars except letters, digits, comma, underscore, brackets, color, hyphen
    label = re.sub(r"[^0-9A-Za-z,_\(\):]", "", label)
    # remove trailing stuff after paranthesis or .
    label = re.sub(r"[\(|:|\.].*", "", label)
    label = re.sub(r"_+$", "", label)
    label = re.sub(r"^_+", "", label)
    labels = re.split(r"_or_|/", label)
    if len(labels) <= 1:
        if len(labels[0]) <= 3 or len(labels[0]) > 30:
            labels[0] = "irrelevant"
        return labels

    ret_labels = []
    for label in labels:
        if label == "":
            continue
        ret_labels += __clean_label(label)
    return ret_labels


# split
def split(
    labels: list[str],
    max_labels: int = 5,
) -> list[str]:
    """
    split is a function that processes labels in a list and provides cleaned up list.
    """

    ret_labels = []

    for label in labels:
        ret_labels += __clean_label(label)

    return ret_labels[:max_labels]


def sanitize(
    subjects: list[str],
    labels: list[str],
) -> Tuple[list[str], list[str]]:
    """
    Sanitize the labels that can be sanitized. models are known sometimes to not answer
    the question or add more details than requested.

    Attributes:
        subjects list[str]: Acceptable subjects in the system.
        labels list[str]: Labels provided by models.

    Return:
        labels list[str]: List of sanitized labels. This list may:
            - be longer than the input
            - may have subjects that are not included in the incoming subjects
        discovered_labels list[str]: List of labels found that are not included in subjects.
    """

    # run through all labels and see if split creates more labels. see testcases.
    row_labels = []
    for label in labels:
        row_labels += split([label])
    labels = row_labels

    # run through every label and fix the ones that startwith a system subject and ignore remaining text.
    row_labels = []
    discovered_labels = []
    for label in labels:
        if label in subjects:
            row_labels.append(label)
            continue

        discovered = True
        for subject in subjects:
            if label.startswith(subject):
                label = subject
                discovered = False
                break

        row_labels.append(label)
        if discovered:
            row_labels.append(label)
            discovered_labels.append(label)

    return row_labels, discovered_labels
