from src.evaluation.metrics import RetrievalMetrics


class TopKActionEvaluator:

    def __init__(
        self,
        retrievers,
        candidate_top_k=(3, 5, 10, 15)
    ):

        self.retrievers = retrievers

        self.candidate_top_k = (
            candidate_top_k
        )

    def evaluate_query(
        self,
        query,
        relevant_scores,
        current_strategy
    ):

        results = []

        retriever = self.retrievers[
            current_strategy
        ]

        original_top_k = getattr(
            retriever,
            "top_k",
            5
        )

        relevant_ids = list(
            relevant_scores.keys()
        )

        try:

            for top_k in (
                self.candidate_top_k
            ):

                retriever.top_k = top_k

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

                    "strategy":
                        current_strategy,

                    "top_k":
                        top_k,

                    "action":
                        (
                            f"set_top_k_{top_k}"
                        ),

                    "ndcg_at_5":
                        ndcg
                })

        finally:

            retriever.top_k = original_top_k

        return results