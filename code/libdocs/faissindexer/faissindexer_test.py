import os
import tempfile

import numpy as np
import pytest
from faissindexer import FaissIndexer, FaissIndexType


@pytest.fixture
def index_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def index_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an indexer and add vectors
        indexer = FaissIndexer(
            index_type=FaissIndexType.brute_force, dimension=128
        )
        vectors = np.random.rand(10, 128).astype("float32")
        ids = np.random.rand(10).astype("int")
        indexer.add(vectors, ids)
        index_file = os.path.join(temp_dir, "test_index.faiss")
        indexer.save_index(index_file)
        yield index_file


@pytest.fixture
def indexer():
    return FaissIndexer(index_type=FaissIndexType.brute_force, dimension=128)


class TestIndexerBasics:
    # Test for FaissIndexType enumeration values
    def test_faiss_index_type_values(self):
        assert FaissIndexType.brute_force == 1
        assert FaissIndexType.hnsw == 2
        assert FaissIndexType.ivf == 3

    # Test the initialization of FaissIndexer
    def test_faiss_indexer_initialization(self, indexer):
        assert indexer.dimension == 128

    # Test adding a single vector to the index
    def test_add_single_vector(self, indexer):
        vector = np.random.rand(1, 128).astype(
            "float32"
        )  # Generating a random vector
        ids = np.random.rand(1).astype("int")
        indexer.add(vector, ids)

    # Test adding multiple vectors to the index
    @pytest.mark.parametrize("num_vectors", [2, 10, 100])
    def test_add_multiple_vectors(self, indexer, num_vectors):
        vectors = np.random.rand(num_vectors, 128).astype(
            "float32"
        )  # Generating multiple vectors
        ids = np.random.rand(num_vectors).astype("int")
        indexer.add(vectors, ids)
        # Verify the vectors have been added appropriately


# Test for index persistence: Saving and Loading
class TestIndexPersistence:
    def test_save_index(self, index_dir):
        # Create an indexer and add vectors
        indexer = FaissIndexer(
            index_type=FaissIndexType.brute_force, dimension=128
        )
        vectors = np.random.rand(10, 128).astype("float32")
        ids = np.random.rand(10).astype("int")
        indexer.add(vectors, ids)

        # Save and verify
        index_file = os.path.join(index_dir, "test_index.faiss")
        indexer.save_index(index_file)
        assert os.path.exists(
            index_file
        ), "Index file should exist after saving."

    def test_load_index(self, index_file):
        indexer = FaissIndexer(
            index_type=FaissIndexType.brute_force, dimension=128
        )
        indexer.load_index(index_file)


# Handling of invalid inputs
class TestInvalidInputs:
    def test_add_invalid_vector_length(self, indexer):
        # Dimensions are 128 in indexer but 200 here.
        with pytest.raises(ValueError, match="mismatched shapes"):
            indexer.add(np.random.rand(1, 200), [1])

    def test_add_invalid_ids_length(self, indexer):
        # More ids than embeddings.
        with pytest.raises(ValueError, match="mismatched embeddings and ids"):
            indexer.add(np.random.rand(1, 128), [1, 1])

    # def test_search_invalid_vector_length(self, indexer):
    #     with pytest.raises(ValueError, match="Query vector length does not match index dimension"):
    #         indexer.search_vectors([np.random.rand(100)], k=1)
