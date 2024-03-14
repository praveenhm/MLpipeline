import re
from typing import List

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


class Chunker:
    """
    Chunker provides capability to provide NLP based chunking capabilities.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        encoder_model_name: str = "distiluse-base-multilingual-cased-v2",
        max_chunk_size: int = 500,
        similarity_threshold: float = 0.2,
    ):
        """
        Constructor for the Chunker class.

        :param model_name: the spacy model to use
        :param encoder_model_name: the encoder model to compute cosine similarity
        :param max_chunk_size: max chunk size for an output chunk
        :param similarity_threshold: similarity threshold for combining sentences
        """
        self.nlp = spacy.load(model_name)
        self.sentence_encoder = SentenceTransformer(encoder_model_name)
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(α, β):
        """
        Compute consine similarity

        :param α: first argument
        :param β: comparison argument
        """
        dot_product = np.dot(α, β)
        norm_α = np.linalg.norm(α)
        norm_β = np.linalg.norm(β)
        return dot_product / (norm_α * norm_β)

    def create_chunks(self, text: str) -> List[str]:
        """
        This method takes a text, cleans it a bit and returns a list of chunks.
        :param text: the text to be chunked
        :return: the chunks as a list
        """
        doc = self.nlp(text)
        # Step 1: Clean the text
        cleaned_text = []
        for ind, sentence in enumerate(doc.sents):
            cleaned = re.sub(r"\s+", " ", sentence.text).strip()
            # Remove lines that are numbers or number.number
            cleaned = re.sub(r"^[0-9\.\s]+$", " ", cleaned)
            # Multi-lines to space
            cleaned = re.sub(r"\n+", " ", cleaned).strip()
            # Multi-space to single space
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 25:
                continue
            cleaned_text.append(cleaned) if (len(cleaned) > 0) else None
        # Step 2: Chunk it to pieces
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
            if current_length + tokens < self.max_chunk_size and (
                previous_sentence is None
                or similarity > self.similarity_threshold
            ):
                text += " " + sentence
                current_length += tokens

            else:
                text = text.strip()
                chunks.append(text) if (len(text) > 0) else None
                text = sentence
                current_length = tokens

            # update the previous sentence
            previous_sentence = current_sentence

        chunks.append(text) if (len(text) > 0) else None
        return chunks

    def chunk_list(self, text_list: List[str]) -> List[List[str]]:
        """
        Chunk a list of strings

        :param text_list: list of strings to be converted into chunks
        """
        output_chunks = []
        for text in text_list:
            chunks = self.create_chunks(text=text)
            output_chunks.append(chunks)
        return output_chunks


if __name__ == "__main__":
    chunker = Chunker(max_chunk_size=10, similarity_threshold=0.1)
    texts = ["This is a test.", "Another test here."]
    chunked_texts = chunker.chunk_list(texts)
