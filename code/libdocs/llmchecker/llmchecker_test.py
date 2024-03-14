import os
import tempfile
from unittest.mock import Mock

import pytest
from libdocs.classifiers.mock.mock import MockModel
from libdocs.types.types import ChunkSubject
from llmchecker import LLMChecker


@pytest.fixture
def mock_model():
    """Provides a mock model that can be used for testing."""
    model = MockModel()
    mock_model = Mock(wraps=model)
    return mock_model


@pytest.fixture
def chunk_subjects():
    """Provides a list of ChunkSubject objects for testing."""
    return [
        ChunkSubject(text="Sample text 1", label="legal"),
        ChunkSubject(text="Sample text 2", label="marketing"),
        ChunkSubject(text="Sample text 3", label="finance"),
        ChunkSubject(text="Sample text 4", label="sales"),
    ]


@pytest.fixture
def llm_checker(mock_model):
    """Provides an LLMChecker instance initialized with a mock model."""
    checker = LLMChecker()
    checker.model = mock_model
    return checker


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestLLMChecker:
    def test_initialization(self, mock_model):
        """Test that the LLMChecker initializes correctly."""
        checker = LLMChecker()
        checker.model = mock_model
        assert checker.model == mock_model
        assert checker.count_correct == 0
        assert checker.count_partial == 0
        assert checker.count_incorrect == 0
        assert checker.data_correct == []
        assert checker.data_partial == []
        assert checker.data_incorret == []

    # def test_run_batch_with_dry_run(self, llm_checker, chunk_subjects):
    #     """Test the run_batch method with dry_run=True."""
    #     results = llm_checker.run_batch(chunks=chunk_subjects, dry_run=True)
    #     # Verify that no classification was performed
    #     assert llm_checker.model.classify.call_count == 0
    #     # Verify that results match the expected dry run output
    #     assert results == ["legal", "marketing"] * len(chunk_subjects)

    # def test_run_batch_classification(self, llm_checker, chunk_subjects):
    #     """Test the run_batch method with actual classification."""
    #     llm_checker.run_batch(chunks=chunk_subjects, dry_run=False)
    #     # Verify that classification was performed
    #     assert llm_checker.model.classify.call_count == 1
    #     # Verify counters are updated correctly based on mock classify return value
    #     assert llm_checker.count_correct == 1
    #     assert llm_checker.count_partial == 1
    #     assert llm_checker.count_incorrect == 1

    # @patch("llmchecker.open", new_callable=mock_open)
    # def test_flush_functionality(self, mock_open, llm_checker, chunk_subjects):
    #     """Test the flush method's file output and reset functionality."""
    #     llm_checker.run_batch(chunks=chunk_subjects, dry_run=False)
    #     # Before flush, verify counts are not zero
    #     assert llm_checker.count_correct > 0
    #     llm_checker._LLMChecker__flush(filebase="test_flush")
    #     # Verify file was attempted to be opened for each category
    #     assert mock_open.call_count == 3
    #     # Verify counters and lists are reset after flush
    #     assert llm_checker.count_correct == 0
    #     assert llm_checker.data_correct == []

    def test_run_with_batch_processing(
        self, llm_checker, chunk_subjects, output_dir
    ):
        """Test the run method for processing chunks in batches."""
        batch_size = 2
        llm_checker.run(
            chunks=chunk_subjects,
            filebase=os.path.join(output_dir, "test_run"),
            batch_size=batch_size,
        )
        # Verify that run_batch was called correctly for batch processing
        assert llm_checker.model.classify.call_count == 2
