from typing import Union

import libdocs.utils.text.text as t
from pandas import DataFrame


class Test:
    def __init__(self, row: t.Row, expected: Union[bool, t.Row]):
        self.row: t.Row = row
        self.expected: Union[bool, t.Row] = expected

    def run(self, i: int, f: Union[t.DropRowFunc, t.MutatingRowFunc]):
        assert (
            f(self.row) == self.expected
        ), f"test {i}: applying {f.__name__}: expected '{self.expected}' for Row {self.row}"


def test_not_all_models_agree():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["b"],
                zephyr_labels=["c"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["b"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["b"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=True,
        ),
    ]
    [test.run(i, t.not_all_models_agree) for (i, test) in enumerate(tests)]


def test_not_model_majority():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["b"],
                zephyr_labels=["c"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["b"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["b"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["b"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
    ]
    [test.run(i, t.not_model_majority) for (i, test) in enumerate(tests)]


def test_not_a_valid_label():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="marketing",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="business_ethics",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
    ]
    [test.run(i, t.not_a_valid_label) for (i, test) in enumerate(tests)]


def test_any_conversation():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["conversation"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["conversation"],
                zephyr_labels=["a"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["conversation"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["conversation"],
                zephyr_labels=["conversation"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
    ]
    [test.run(i, t.any_conversation) for (i, test) in enumerate(tests)]


def test_any_irrelevant():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["irrelevant"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["irrelevant"],
                zephyr_labels=["a"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["irrelevant"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["irrelevant"],
                zephyr_labels=["irrelevant"],
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="test",
                entity_id="sha256sum",
                input_src="test",
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=False,
        ),
    ]
    [test.run(i, t.any_irrelevant) for (i, test) in enumerate(tests)]


def test_short_sentences():
    tests = [
        Test(
            row=t.Row(
                text="One",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="One Two",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="Still too short.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="However, this sentence is now a bit longer than the average 15 words of an English sentence.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
    ]
    [test.run(i, t.short_sentences) for (i, test) in enumerate(tests)]


def test_contains_url():
    tests = [
        Test(
            row=t.Row(
                text="This chunk does not contain a URL.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="This chunk does not contains a URL to http://www.kernel.org/",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="Books sometimes have weird URLs like HTTP://WWW.GOOGLE.COM/",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="Or even more weird, they capitalize them like here: Http://www.kernel.org/",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
    ]
    [test.run(i, t.contains_url) for (i, test) in enumerate(tests)]


def test_text_starts_with_alpha():
    tests = [
        Test(
            row=t.Row(
                text="42 might be the answer to all questions, but this will still get filtered.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="Fourty-two, however, is completely different.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
    ]
    [test.run(i, t.text_starts_with_alpha) for (i, test) in enumerate(tests)]


def test_contains_chapter_reference():
    tests = [
        Test(
            row=t.Row(
                text="Chapter 1: Introduction",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="CHAPTER 1: Introduction",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="As seen in the introduction (chapter 1), we are going to do something with this here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="This does not contain a reference to a classic book content separator.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
    ]
    [
        test.run(i, t.contains_chapter_reference)
        for (i, test) in enumerate(tests)
    ]


def test_contains_year_reference():
    tests = [
        Test(
            row=t.Row(
                text="As can be seen in Davidson (2016) causality is a corner stone of philosophy of action, and its biggest problem.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="As can be seen in Davidson [2016] causality is a corner stone of philosophy of action, and its biggest problem.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="As can be seen in Hume (1962) humanity is forfeit to achieve greatness based on logic.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="Previously, we defined a large number A (3198).",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="Previously, we defined a large number A (191988).",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="This does not contain a reference to a year.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
    ]
    [test.run(i, t.contains_year_reference) for (i, test) in enumerate(tests)]


def test_is_diagram_description():
    tests = [
        Test(
            row=t.Row(
                text="This is not a diagram description.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="Figure 4.3: We are seeing a diagram description here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="figure (a): We are seeing another diagram description here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="FIGURE 42: We are seeing yet another diagram description here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
    ]
    [test.run(i, t.is_diagram_description) for (i, test) in enumerate(tests)]


def test_is_table_description():
    tests = [
        Test(
            row=t.Row(
                text="This is not a table description.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=False,
        ),
        Test(
            row=t.Row(
                text="Table 4.3: We are seeing a diagram description here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="table (a): We are seeing another diagram description here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
        Test(
            row=t.Row(
                text="TABLE 42: We are seeing yet another diagram description here.",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=True,
        ),
    ]
    [test.run(i, t.is_table_description) for (i, test) in enumerate(tests)]


def test_all_models_disagreeing_with_input():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="input_got_it_wrong",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                mistral_labels=["a"],
                zephyr_labels=["a"],
            ),
            expected=t.Row(
                text="unimportant",
                label="a",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="input_incorrect",
                mistral_labels=["a"],
                mistral_verdict="input_incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="input_incorrect",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="input_got_it_wrong",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="a",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="input_incorrect",
                mistral_labels=["a"],
                mistral_verdict="input_incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="input_incorrect",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="c",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["b"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="c",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["b"],
                zephyr_verdict="incorrect",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="c",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["b"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="c",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["b"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="c",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["b"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="c",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["b"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                debert_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
        ),
    ]
    [
        test.run(i, t.all_models_disagreeing_with_input)
        for (i, test) in enumerate(tests)
    ]


def test_ensure_only_ascii_characters():
    tests = [
        Test(
            row=t.Row(
                text="This sentence contains only ASCII characters",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=t.Row(
                text="This sentence contains only ASCII characters",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
        ),
        Test(
            row=t.Row(
                text="– This sentence contains not only ASCII characters: we estimated W₂W₁ as follows",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
            expected=t.Row(
                text="This sentence contains not only ASCII characters: we estimated WW as follows",
                label="test",
                entity_id="sha256sum",
                input_src="test",
            ),
        ),
    ]
    [
        test.run(i, t.ensure_only_ascii_characters)
        for (i, test) in enumerate(tests)
    ]


def test_reverse_labeling():
    tests = [
        Test(
            row=t.Row(
                text="unimportant",
                label="b",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="b",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="a",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="n/a",
                mistral_labels=["a"],
                mistral_verdict="n/a",
                zephyr_labels=["a"],
                zephyr_verdict="n/a",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["b"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="a",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="n/a",
                mistral_labels=["a"],
                mistral_verdict="n/a",
                zephyr_labels=["b"],
                zephyr_verdict="n/a",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="incorrect",
                mistral_labels=["b"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="a",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="n/a",
                mistral_labels=["b"],
                mistral_verdict="n/a",
                zephyr_labels=["a"],
                zephyr_verdict="n/a",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["b"],
                deberta_verdict="incorrect",
                mistral_labels=["a"],
                mistral_verdict="incorrect",
                zephyr_labels=["a"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="a",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["b"],
                deberta_verdict="n/a",
                mistral_labels=["a"],
                mistral_verdict="n/a",
                zephyr_labels=["a"],
                zephyr_verdict="n/a",
            ),
        ),
        Test(
            row=t.Row(
                text="unimportant",
                label="unlabeled",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="incorrect",
                mistral_labels=["b"],
                mistral_verdict="incorrect",
                zephyr_labels=["c"],
                zephyr_verdict="incorrect",
            ),
            expected=t.Row(
                text="unimportant",
                label="b",
                entity_id="sha256sum",
                input_src="test",
                deberta_labels=["a"],
                deberta_verdict="n/a",
                mistral_labels=["b"],
                mistral_verdict="n/a",
                zephyr_labels=["c"],
                zephyr_verdict="n/a",
            ),
        ),
    ]
    [test.run(i, t.reverse_labeling) for (i, test) in enumerate(tests)]


def test_dedup_input_data():
    data = [
        {
            "text": "A duplicate piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text",
            "entity_id": "b",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text 2",
            "entity_id": "c",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text 3",
            "entity_id": "d",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "A duplicate piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text 3",
            "entity_id": "d",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "A duplicate piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
    ]
    dedup_data = [
        {
            "text": "A duplicate piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text",
            "entity_id": "b",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text 2",
            "entity_id": "c",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "another piece of text 3",
            "entity_id": "d",
            "input_src": "test",
            "label": "a",
        },
    ]
    should_be_df = DataFrame(dedup_data)

    input_df = DataFrame(data)
    t.dedup_input_data(input_df)

    assert input_df.equals(should_be_df)


def test_drop_input_data():
    data = [
        {
            "text": "A piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "42 and another piece of text",
            "entity_id": "b",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "2738.34 another piece of text",
            "entity_id": "c",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "- and another piece of text",
            "entity_id": "d",
            "input_src": "test",
            "label": "a",
        },
    ]
    after_drop_data = [
        {
            "text": "A piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
    ]
    should_be_df = DataFrame(after_drop_data)

    in_df = DataFrame(data)
    out_df = t.drop_input_data(in_df, [t.text_starts_with_alpha])

    assert out_df.equals(should_be_df)


def test_mutate_input_data():
    data = [
        {
            "text": "A piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "₂ W₂W₁ and another piece of text",
            "entity_id": "b",
            "input_src": "test",
            "label": "a",
        },
    ]
    after_drop_data = [
        {
            "text": "A piece of text",
            "entity_id": "a",
            "input_src": "test",
            "label": "a",
        },
        {
            "text": "WW and another piece of text",
            "entity_id": "b",
            "input_src": "test",
            "label": "a",
        },
    ]
    should_be_df = DataFrame(after_drop_data)

    in_df = DataFrame(data)
    out_df = t.mutate_input_data(in_df, [t.ensure_only_ascii_characters])

    assert out_df.equals(should_be_df)
