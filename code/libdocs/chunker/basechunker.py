import re
from abc import ABC, abstractmethod
# import itertools
from typing import List

import spacy
from sentence_transformers import SentenceTransformer

# control_chars = "".join(map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0))))
# control_char_re = re.compile("[%s]" % re.escape(control_chars))


class BaseChunker(ABC):

    def __init__(
        self,
        sentence_encoder_model: str = "distiluse-base-multilingual-cased-v2",
        spacy_model: str = "en_core_web_sm",
        min_chunk_size: int = 80,
    ):
        """
        Initializes the base chunker with desired models and parameters.

        Args:
            sentence_encoder_model (str): The sentence encoder model name.
            spacy_model (str): The spaCy model name.
            min_chunk_size (int): The minimum chunk size.
        """
        super().__init__()
        self.sentence_encoder = SentenceTransformer(sentence_encoder_model)
        self.min_chunk_size = min_chunk_size

        self.nlp = self._init_nlp(spacy_model)

    @classmethod
    def chunker_name(cls):
        return cls.__name__.lower()

    @staticmethod
    def _init_nlp(spacy_model: str):
        nlp = spacy.load(
            spacy_model,
            disable=[
                "ner",
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        nlp.add_pipe("sentencizer")
        return nlp

    @abstractmethod
    def create_chunks(self, text: str) -> list[str]:
        pass

    def process_text(self, text: str) -> List[str]:
        """
        Processes a chunk of text based on the specified strategy and writes the chunks to an output file.

        Args:
            chunk (str): The input text chunk.
            output_file (str): Output file path.
            separator (str): Handling separators in file (if it there).

        Returns:
            List[str] : List of the chunks
        """
        text = self.clean_text(text)
        chunks = self.create_chunks(text)

        return chunks

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing newlines, tabs, and extra spaces within sentences.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\n+", " ", cleaned).strip()

        return cleaned
