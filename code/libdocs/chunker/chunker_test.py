import numpy as np
import pytest
from chunker import Chunker


class TestChunkerBasics:
    def test_initialization(self):
        # Default initialization
        chunker = Chunker()
        assert chunker.max_chunk_size == 500
        assert chunker.similarity_threshold == 0.2

        # Custom initialization
        custom_chunker = Chunker(max_chunk_size=300, similarity_threshold=0.3)
        assert custom_chunker.max_chunk_size == 300
        assert custom_chunker.similarity_threshold == 0.3

    def test_cosine_similarity(self):
        vector_a = np.array([1, 0, 0])
        vector_b = np.array([0, 1, 0])
        vector_c = np.array([1, 0, 0])

        # Orthogonal vectors
        assert Chunker.cosine_similarity(vector_a, vector_b) == pytest.approx(
            0.0
        )

        # Identical vectors
        assert Chunker.cosine_similarity(vector_a, vector_c) == pytest.approx(
            1.0
        )


class TestChunkerChunking:
    def test_chunk_list(self):
        chunker = Chunker(max_chunk_size=10, similarity_threshold=0.1)
        texts = [
            "This is a test but it must have more than 25 chars as a limit to do chunking.",
            "Another test here but it must have more than 25 chars as a limit to do chunking.",
        ]
        chunked_texts = chunker.chunk_list(texts)

        assert len(chunked_texts) == 2
        assert all(isinstance(chunks, list) for chunks in chunked_texts)
        print(chunked_texts)
        assert all(len(chunks) > 0 for chunks in chunked_texts)

    def test_chunk(self):
        chunker = Chunker(max_chunk_size=12, similarity_threshold=0.1)
        text = "This is a test. This test is longer than the max_chunk_size, requiring splitting. Adding another sentence which must cause a split."

        chunks = chunker.create_chunks(text)

        # Ensure the text is split into multiple chunks
        assert isinstance(chunks, list)
        assert len(chunks) > 1

        # Each chunk should not exceed the max_chunk_size in terms of tokens
        for chunk in chunks:
            assert len(chunker.nlp(chunk)) <= chunker.max_chunk_size
