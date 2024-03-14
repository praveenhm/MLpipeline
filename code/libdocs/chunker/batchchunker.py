import math
from typing import List

import numpy as np
from scipy.signal import argrelextrema
from sentence_transformers import util

from .basechunker import BaseChunker


class BatchChunker(BaseChunker):
    def __init__(self):
        super().__init__()

    def create_chunks(self, text: str) -> list[str]:
        sentence = self.batch_sentencize(text)
        embeddings = self.batch_embed(sentence)
        chunks = self.chunk_embeddings(embeddings, sentence)
        return chunks

    def batch_sentencize(self, text: str) -> List[str]:
        """
        Applies sentence tokenization to a batch of documents.

        Args:
            documents (List[str]): List of documents.

        Returns:
            List[str]: List of tokenized sentences for each document.
        """
        docs = self.nlp.pipe([text], batch_size=16)
        return self.sentencize(list(docs)[0])

    def batch_embed(self, sentences: List[str]) -> List[np.array]:
        """
        Applies sentence embedding to a batch of sentences.

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            np.array: Array of sentence embeddings.
        """
        embeddings = self.sentence_encoder.encode(
            sentences,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def chunk_embeddings(
        self, embeddings: np.array, sentences: List[str]
    ) -> List[str]:
        """
        Divides the input text chunk into chunks based on sentence embeddings.

        Args:
            embeddings (np.array): Array of sentence embeddings.
            sentences (List[str]): List of sentences.

        Returns:
            List[str]: List of text chunks.
        """
        embeddings = np.stack(embeddings)
        split_points = self.get_split_points(embeddings)
        chunks = self.get_chunks(sentences, split_points)
        return chunks

    def sentencize(self, doc) -> List[str]:
        """
        Tokenizes sentences in a document.

        Args:
            doc: spaCy document.

        Returns:
            List[str]: List of tokenized sentences.
        """
        sents = []
        for sent in doc.sents:
            sents.append(sent.text)
        return sents

    def get_split_points(self, embeddings: np.array):
        """
        Determines split points in the text chunk based on sentence embeddings.

        Args:
            embeddings (np.array): Array of sentence embeddings.

        Returns:
            Set[int]: Set of split points.
        """
        similarities = util.cos_sim(embeddings, embeddings)
        activated_similarities = self.activate_similarities(
            similarities, p_size=10
        )
        minima = argrelextrema(activated_similarities, np.less, order=1)
        split_points = {each for each in minima[0]}
        return split_points

    def get_chunks(self, sents: List[str], points) -> List[str]:
        """
        Divides the input sentences into chunks based on split points.

        Args:
            sents (List[str]): List of sentences.
            points (Set[int]): Set of split points.

        Returns:
            List[str]: List of text chunks.
        """
        if len(points) == 0:
            return [" ".join(sents)]

        chunks = []
        current_chunk = []
        for idx, sent in enumerate(sents):
            if idx in points:
                current_chunk.append(sent)
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            else:
                current_chunk.append(sent)

        if len(current_chunk) > self.min_chunk_size:
            chunks.append(" ".join(current_chunk))

        return chunks

    def activate_similarities(
        self, similarities: np.array, p_size: int = 10
    ) -> np.array:
        """
        Activates similarities using a sigmoid function.

        Args:
            similarities (np.array): Array of similarities.
            p_size (int, optional): Size of the activation function. Defaults to 10.

        Returns:
            np.array: Activated similarities.
        """
        p_size = min(p_size, similarities.shape[0])
        x = np.linspace(-10, 10, p_size)
        y = np.vectorize(self.rev_sigmoid)
        activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
        diagonals = [
            similarities.diagonal(each)
            for each in range(0, similarities.shape[0])
        ]
        diagonals = [
            np.pad(each, (0, similarities.shape[0] - len(each)))
            for each in diagonals
        ]
        diagonals = np.stack(diagonals)
        diagonals = diagonals * activation_weights.reshape(-1, 1)
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

    def rev_sigmoid(self, x: float) -> float:
        """
        Computes the reverse sigmoid function.

        Args:
            x (float): Input value.

        Returns:
            float: Reverse sigmoid output.
        """
        return 1 / (1 + math.exp(0.5 * x))
