import logging
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..types.types import LabeledChunk


class Embedder:
    """
    Embeds text chunks using a specified model.

    Args:
        model_name (str): The name of the sentence embedding model to use.
        model_path (str, optional): The path to the pre-trained model, if loading from a file.
    """

    def __init__(
        self,
        model_name: str = "distiluse-base-multilingual-cased-v2",
        index_filepath: str = None,
    ):
        """
        Initializes the Embedder class.

        Args:
            model_name (str, optional): The name of the sentence embedding model to use.:
        """
        self.gpu = torch.cuda.is_available()
        self.__model = SentenceTransformer(model_name)
        # GPU support
        if self.gpu:
            logging.info("activating gpu for faiss")
            self.__model = self.__model.to("cuda")
        self.chunks = None
        self.embeddings = None
        if index_filepath is not None:
            self.load_embeddings(index_filepath)

    def get_dimensions(self):
        return self.__model.get_sentence_embedding_dimension()

    def encode(
        self, texts: list[str], show_progress_bar: bool = False
    ) -> list[np.ndarray]:
        """
        Encode a set of texts into vectors with GPU enhancements.

        Args:
            texts (list[str]): A list of texts snippets.

        Returns:
            list[np.ndarray]: A list of NumPy arrays, each representing the embedding of a corresponding chunk.
        """
        embeddings = self.__model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return embeddings

    def create_embedding(self, chunks: list[LabeledChunk]) -> list[np.ndarray]:
        """
        Embeds a list of LabeledChunk objects.

        Args:
            chunks (list[LabeledChunk]): A list of LabeledChunk objects containing text snippets.

        Returns:
            list[np.ndarray]: A list of NumPy arrays, each representing the embedding of a corresponding chunk.
        """
        texts = [chunk.text for chunk in chunks]
        self.chunks = chunks
        self.embeddings = self.encode(texts, show_progress_bar=True)
        assert len(self.embeddings) == len(
            self.chunks
        ), f"embeddings: {len(self.embeddings)} chunks: {len(self.chunks)}"
        return self.embeddings

    def save_embeddings(self, index_filepath: str):
        # store the labeled chunks
        chunk_index = index_filepath
        chunk_index = chunk_index.replace(".faiss", ".chunks.pkl")
        file = open(chunk_index, "wb")
        pickle.dump(self.chunks, file)
        file.close()

        # store the embeddings
        embed_index = index_filepath
        embed_index = embed_index.replace(".faiss", ".embeds.pkl")
        file = open(embed_index, "wb")
        pickle.dump(self.embeddings, file)
        file.close()

    def load_embeddings(self, index_filepath: str):
        # read labeled chunks
        chunk_index = index_filepath
        chunk_index = chunk_index.replace(".faiss", ".chunks.pkl")
        if os.path.isfile(chunk_index):
            logging.info(f"Loading chunks from {chunk_index}")
            file = open(chunk_index, "rb")
            chunks = pickle.load(file)
            file.close()
            self.chunks = chunks

        # read the embeddings
        embed_index = index_filepath
        embed_index = embed_index.replace(".faiss", ".embeds.pkl")
        if os.path.isfile(embed_index):
            logging.info(f"Loading embeddings from {embed_index}")
            file = open(embed_index, "rb")
            self.embeddings = pickle.load(file)
            file.close()

        assert len(self.embeddings) == len(
            self.chunks
        ), f"embeddings: {len(self.embeddings)} chunks: {len(self.chunks)}"
