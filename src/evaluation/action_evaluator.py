from src.evaluation.metrics import RetrievalMetrics


class ActionEvaluator:

    def __init__(
        self,
        retrievers
    ):

        self.retrievers = retrievers

    def evaluate_query(
        self,
        query,
        relevant_scores,
        query_type
    ):

        relevant_ids = list(
            relevant_scores.keys()
        )

        results = []

        for strategy, retriever in (
            self.retrievers.items()
        ):

            result = retriever.retrieve(
                query
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk
                in result.retrieved_chunks
            ]

            ndcg = (
                RetrievalMetrics.ndcg_at_k(
                    retrieved_ids,
                    relevant_scores,
                    5
                )
            )

            results.append({
                "query":
                    query,

                "query_type":
                    query_type,

                "strategy":
                    strategy,

                "ndcg_at_5":
                    ndcg,

                "retrieved_ids":
                    retrieved_ids
            })

        return results