from src.evaluation.metrics import RetrievalMetrics


class TopKActionEvaluator:

    def __init__(
        self,
        retrievers,
        candidate_top_k=(
            3,
            5,
            10,
            15
        ),
        cost_weight: float = 0.10
    ):

        self.retrievers = retrievers

        self.candidate_top_k = tuple(
            sorted(candidate_top_k)
        )

        self.cost_weight = (
            cost_weight
        )

    def evaluate_query(
        self,
        query,
        relevant_scores,
        current_strategy,
        current_top_k=5
    ):

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

        candidate_top_k = [
            top_k
            for top_k in self.candidate_top_k
            if top_k > current_top_k
        ]

        results = []

        try:

            retriever.top_k = (
                current_top_k
            )

            current_result = (
                retriever.retrieve(
                    query
                )
            )

            current_ids = [
                chunk.chunk_id
                for chunk in (
                    current_result.retrieved_chunks
                )
            ]

            current_recall = (
                RetrievalMetrics.recall_at_k(
                    current_ids,
                    relevant_ids,
                    current_top_k
                )
            )

            current_ndcg = (
                RetrievalMetrics.ndcg_at_k(
                    current_ids,
                    relevant_scores,
                    current_top_k
                )
            )

            current_utility = (
                current_ndcg
            )

            results.append({
                "query": query,
                "strategy": current_strategy,
                "current_top_k": current_top_k,
                "top_k": current_top_k,
                "action": "keep",
                "recall_at_k": current_recall,
                "ndcg_at_k": current_ndcg,
                "incremental_recall": 0.0,
                "additional_retrieval": 0,
                "cost_ratio": 1.0,
                "cost_penalty": 0.0,
                "utility": current_utility,
                "current_utility": current_utility,
                "utility_gain": 0.0
            })

            for top_k in candidate_top_k:

                retriever.top_k = top_k

                result = (
                    retriever.retrieve(
                        query
                    )
                )

                retrieved_ids = [
                    chunk.chunk_id
                    for chunk in (
                        result.retrieved_chunks
                    )
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

                incremental_recall = max(
                    0.0,
                    recall - current_recall
                )

                cost_ratio = (
                    top_k
                    /
                    current_top_k
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

                utility_gain = (
                    utility
                    -
                    current_utility
                )

                results.append({
                    "query": query,
                    "strategy": current_strategy,
                    "current_top_k": current_top_k,
                    "top_k": top_k,
                    "action": (
                        f"set_top_k_{top_k}"
                    ),
                    "recall_at_k": recall,
                    "ndcg_at_k": ndcg,
                    "incremental_recall": (
                        incremental_recall
                    ),
                    "additional_retrieval": (
                        top_k - current_top_k
                    ),
                    "cost_ratio": cost_ratio,
                    "cost_penalty": (
                        cost_penalty
                    ),
                    "utility": utility,
                    "current_utility": (
                        current_utility
                    ),
                    "utility_gain": (
                        utility_gain
                    )
                })

        finally:

            retriever.top_k = (
                original_top_k
            )

        return results