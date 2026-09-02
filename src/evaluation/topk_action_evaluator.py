from src.evaluation.metrics import RetrievalMetrics


class TopKActionEvaluator:

    def __init__(
        self,
        retrievers,
        candidate_top_k=(3, 5, 10, 15),
        cost_weight: float = 0.10
    ):

        self.retrievers = retrievers

        self.candidate_top_k = (
            candidate_top_k
        )

        self.cost_weight = (
            cost_weight
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

                recall = (
                    RetrievalMetrics.recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        top_k
                    )
                )

                ndcg = (
                    RetrievalMetrics.ndcg_at_k(
                        retrieved_ids,
                        relevant_scores,
                        top_k
                    )
                )

                cost_ratio = (
                    top_k / original_top_k
                )

                cost_penalty = (
                    self.cost_weight
                    *
                    max(
                        0.0,
                        cost_ratio - 1.0
                    )
                )

                utility = (
                    ndcg
                    -
                    cost_penalty
                )

                results.append({
                    "query":
                        query,

                    "strategy":
                        current_strategy,

                    "top_k":
                        top_k,

                    "action":
                        f"set_top_k_{top_k}",

                    "recall_at_k":
                        recall,

                    "ndcg_at_k":
                        ndcg,

                    "cost_ratio":
                        cost_ratio,

                    "cost_penalty":
                        cost_penalty,

                    "utility":
                        utility
                })

        finally:

            retriever.top_k = original_top_k

        return results