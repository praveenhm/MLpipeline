import re

import numpy as np

from .basechunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    This class provides the functionality of chunking text into smaller pieces,
     after some cleanup. The cleanup includes removing newlines,
     tabs, and extra spaces within a sentence, etc.
    """

    def __init__(
        self, chunk_size: int = 500, similarity_threshold: float = 0.2
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def create_list_of_chunks(self, text_list: list[str]) -> list[list[str]]:
        output_chunks = []
        for text in text_list:
            output_chunks.append(self.create_chunks(text=text))
        return output_chunks

    def create_chunks(self, text: str) -> list[str]:
        """
        This method takes a text, cleans it a bit and returns a list of chunks.
        :param text: the text to be chunked
        :return: the chunks as a list
        """
        doc = self.nlp(text)

        cleaned_text = []
        for ind, sentence in enumerate(doc.sents):
            cleaned = re.sub(r"\s+", " ", sentence.text).strip()
            cleaned = re.sub(r"\n+", " ", cleaned).strip()
            cleaned_text.append(cleaned) if (len(cleaned) > 0) else None

        chunks = []
        text = ""
        current_length = 0
        previous_sentence = None
        for sentence in cleaned_text:
            tokens = len(self.nlp(sentence))
            current_sentence = self.sentence_encoder.encode(sentence)
            similarity = 1
            if previous_sentence is not None:
                similarity = self.cosine_similarity(
                    previous_sentence, current_sentence
                )
            if current_length + tokens < self.chunk_size and (
                previous_sentence is None
                or similarity > self.similarity_threshold
            ):
                text += " " + sentence
                current_length += tokens

            else:
                text = text.strip()
                (
                    chunks.append(text)
                    if (len(text) > self.min_chunk_size)
                    else None
                )
                text = sentence
                current_length = tokens

            # update the previous sentence
            previous_sentence = current_sentence

        chunks.append(text) if (len(text) > self.min_chunk_size) else None
        return chunks
