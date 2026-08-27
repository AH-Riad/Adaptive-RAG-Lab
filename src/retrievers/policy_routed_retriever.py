from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.planning.calibrated_policy import CalibratedPolicy


class PolicyRoutedRetriever(BaseRetriever):

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        hybrid_retriever,
        policy_path: str
    ):
        self.retrievers = {
            "dense": dense_retriever,
            "bm25": bm25_retriever,
            "hybrid": hybrid_retriever
        }

        self.policy = CalibratedPolicy(
            path=policy_path
        )

    def retrieve(
        self,
        query: str
    ) -> RetrievalResult:

        analysis = self._analyze_query(
            query
        )

        query_type = analysis[
            "query_type"
        ]

        strategy = (
            self.policy.get_strategy(
                query_type
            )
        )

        retriever = self.retrievers[
            strategy.value
        ]

        result = retriever.retrieve(
            query
        )

        return result

    def _analyze_query(
        self,
        query: str
    ):

        from src.analyzer.query_analyzer import (
            QueryAnalyzer
        )

        analyzer = QueryAnalyzer()

        return analyzer.analyze(
            query
        )