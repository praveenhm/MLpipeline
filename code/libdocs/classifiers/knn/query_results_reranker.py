import logging as log
from typing import List, Tuple

from sentence_transformers import CrossEncoder


class QueryResultsReRanker:
    """
    This class provides the functionality of chunking text into smaller pieces,
     after some cleanup. The cleanup includes removing newlines,
     tabs, and extra spaces within a sentence, etc.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "cross-encoder/stsb-distilroberta-base"
        log.info(
            f"Configuration specifies the cross encoder model: "
            f"{self.model_name}. Loading it..."
        )
        try:
            self.model = CrossEncoder(self.model_name)
            log.info(f"Loaded cross encoder model: {self.model_name}")
        except Exception as e:
            log.error(f"Error while loading cross encoder model: {e}")
            raise e

    def rerank(
        self, query: str, results: List[Tuple[int, str]]
    ) -> List[Tuple[int, str, float]]:
        """
        This method takes a query and a list of results, and returns a re-ranked
        list of results reverse-sorted by the scores determined by the model.
        :param query: the query text
        :param results: the list of results and their scores.
        """

        if query is None or len(query) == 0:
            log.error(f"Query cannot be empty: {query}")
            raise ValueError("Query cannot be empty")
        if results is None or len(results) == 0:
            log.error(f"Candidate results cannot be empty: {results}")
            raise ValueError("Candidates cannot be empty")
        try:
            # Use the model to compute scores for each result in relation to the query
            scores = self.model.predict(
                [[query, result[1]] for result in results]
            )
            # Sort the results by their scores
            sorted_results = sorted(
                zip(results, scores), key=lambda x: x[1], reverse=True
            )
            sorted_result_tuples = [
                (result[0][0], result[0][1], float(result[1]))
                for result in sorted_results
            ]
            return sorted_result_tuples
        except Exception as e:
            log.error(
                f"Model inference error while reranking results: {e}. "
                f"The query was: {query}. The results were: {results}"
            )
            raise e
