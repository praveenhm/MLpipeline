import logging
import math
import time

import numpy as np
from libdocs.embedder.embedder import Embedder
from libdocs.faissindexer.faissindexer import FaissIndexer

from .query_results_reranker import QueryResultsReRanker
from .type import (ClassifierPrediction, ClassifierPredictionError,
                   CrossEncoderChunkScore, LabeledChunk,
                   LabeledChunkPrediction)


class KnnEmbeddingClassifier:
    def __init__(
        self,
        index_file: str,
        use_kernelization: bool = True,
        cross_encode: bool = False,
        model_name: str = "distiluse-base-multilingual-cased-v2",
    ):
        self.faiss_indexer = None
        self.embedder = None
        self.use_kernelization = use_kernelization
        self.model_name = model_name
        self.cross_encode = cross_encode
        if cross_encode:
            self.reranker = QueryResultsReRanker()
        else:
            self.reranker = None

        self.load_faiss_index(index_file)
        length = len(self.chunk_id_map)
        self.top_neighbors = 100
        self.top_predictions = math.ceil(
            pow(length, 1.0 / 3)
        )  # To take top k as the cube root of the number of chunks

    def softmax(
        self,
        x,
    ):
        """Compute softmax values for each element in x."""
        e_x = np.exp(
            x - np.max(x)
        )  # Subtracting np.max(x) for numerical stability
        return e_x / e_x.sum()

    def load_faiss_index(
        self,
        index_file: str,
    ):
        """
        Load the index-mapped Faiss index
        """
        if not index_file:
            raise ValueError(
                f"No index file specified in the path: {index_file}"
            )

        embedding_dimension = 512
        self.faiss_indexer = FaissIndexer(
            index_file=index_file, dimension=embedding_dimension
        )
        self.embedder = Embedder(self.model_name, index_file)
        self.labeled_chunks = self.embedder.chunks
        self.create_chunk_id_map()

    def create_chunk_id_map(self) -> None:
        """
        Creates a lookup map for the labeled chunks from its id.
        """
        self.chunk_id_map = {chunk.id: chunk for chunk in self.labeled_chunks}

    def predict(
        self,
        X: list[str],
        top: int,
    ) -> list[list[ClassifierPrediction]]:
        """
        Predict the class label for the given text.
        """

        # Pre-conditions
        if not X:
            raise ValueError(
                'Must provide a non-empty value for "X" parameter.'
            )

        # Create an embedding vector out of the query text
        start_time = time.time()
        query_vectors = self.embedder.encode(
            X,
            show_progress_bar=True,
        )  # input is a list of strings
        logging.info(
            f"encoding time: {time.time()-start_time} for {len(X)} entries"
        )

        # Use the Faiss index to find the k-nearest neighbors
        start_time = time.time()
        matches, indices = self.faiss_indexer.index.search(
            query_vectors, self.top_neighbors
        )
        logging.info(
            f"search time: {time.time()-start_time} for {len(query_vectors)} vectors"
        )

        assert len(matches) == len(
            indices
        ), f"matches: {len(matches)} indices:{len(indices)}"

        start_time = time.time()
        list_of_predictions: list[list[ClassifierPrediction]] = []
        for index, match in enumerate(matches):
            # Look-up the text of the k-nearest neighbors
            neighbors = []
            for id in indices[index]:
                t = self.chunk_id_map[id]
                t.id = id
                neighbors.append(t)

            if self.cross_encode:
                # With Cross Encoder
                tuples = [
                    [neighbor.id, neighbor.text] for neighbor in neighbors
                ]
                scored_tuples = self.reranker.rerank(X, tuples)

                chunk_scores = []
                for tuple in scored_tuples:
                    id = tuple[0]
                    labeled_chunk = self.chunk_id_map[id]
                    score = tuple[2]
                    chunk_score = CrossEncoderChunkScore(
                        label_chunk=labeled_chunk, score=score
                    )
                    chunk_scores.append(chunk_score)
            else:
                # Without Cross Encoder
                chunk_scores = []
                for i, neighbor in enumerate(neighbors):
                    score = 100000000
                    if match[i]:
                        score = 1 / match[i]
                    chunk_scores.append(
                        CrossEncoderChunkScore(
                            label_chunk=neighbor, score=score
                        )
                    )

            # Derive subject predictions from the chunk scores
            predictions = self.to_predictions(chunk_scores)
            list_of_predictions.append(predictions[:top])
        logging.info(
            f"predict time: {time.time()-start_time} for {len(list_of_predictions)} predictions"
        )

        return list_of_predictions

    def to_predictions(
        self,
        chunk_scores: list[CrossEncoderChunkScore],
    ) -> list[ClassifierPrediction]:
        """
        Convert the chunk scores to predictions.
        """
        # 0. Accumulate the scores of scores by subject
        subject_scores = {}
        for chunk_score in chunk_scores:
            subject = chunk_score.label_chunk.subject
            score = 1
            if self.use_kernelization:
                score = chunk_score.score
            if subject in subject_scores:
                subject_scores[subject] += score
            else:
                subject_scores[subject] = score

        # 1. Softmax these votes to get the class probabilities
        key_values = subject_scores.items()
        values = [key_value[1] for key_value in key_values]
        softmax_values = self.softmax(values)
        softmax_values = softmax_values.tolist()

        # 2. For each class, create the ClassifierPrediction object
        predictions = []
        for i, key_value in enumerate(key_values):
            subject = key_value[0]
            score = softmax_values[i]
            prediction = ClassifierPrediction(y_pred=subject, y_proba=score)
            predictions.append(prediction)

        predictions.sort(key=lambda x: x.y_proba, reverse=True)
        return predictions

    def summary_evaluation(
        self,
        test_chunks: list[LabeledChunk],
        top: int,
    ) -> None:
        """
        Print a summary of the evaluation of the classifier.
        """
        # Pre-conditions
        if not test_chunks:
            raise ValueError(
                'Must provide a non-empty value for "test_chunks" parameter.'
            )

        test_chunk_predictions: list[list[LabeledChunkPrediction]] = (
            self.chunks_to_predictions(test_chunks, top)
        )

        # Next, let us compute the overall accuracy
        total = len(test_chunk_predictions)

        errors = [
            test_chunk_prediction
            for test_chunk_prediction in test_chunk_predictions
            if test_chunk_prediction.top_prediction_error is not None
        ]
        total_errors = len(errors)

        accuracy = 1.0 * (total - total_errors) / total

        return accuracy

    def chunks_to_predictions(
        self,
        labeled_chunks: list[LabeledChunk],
        top: int,
    ) -> list[LabeledChunkPrediction]:

        X = []
        y = []
        for labeled_chunk in labeled_chunks:
            X.append(labeled_chunk.text)
            y.append(labeled_chunk.subject)

        top_predictions = self.predict(X, top)

        assert len(X) == len(y), f"X: {len(X)} y:{len(y)}"
        assert len(X) == len(
            labeled_chunks
        ), f"X: {len(X)} labeled_chunks:{len(labeled_chunks)}"
        assert len(X) == len(
            top_predictions
        ), f"X: {len(X)} top_predictions:{len(top_predictions)}"

        predictions = []
        for index, top_prediction in enumerate(top_predictions):
            top_predictions_labels = [
                top.y_pred.lower().strip() for top in top_prediction
            ]

            prediction_error = None

            if y[index].lower().strip() not in top_predictions_labels:
                error_message = f"Incorrect prediction: {top_predictions_labels}, correct label: {y[index]}"
                prediction_error = ClassifierPredictionError(
                    error=error_message,
                    top_predictions=top_prediction,
                    ground_truth=y[index],
                )

            predictions.append(
                LabeledChunkPrediction(
                    labeled_chunk=labeled_chunks[index],
                    top_predictions=top_prediction,
                    top_prediction_error=prediction_error,
                )
            )

        return predictions
