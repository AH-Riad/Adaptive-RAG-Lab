from src.evaluation.metrics import RetrievalMetrics


class ActionEvaluator:

    STRATEGIES = [
        "dense",
        "bm25",
        "hybrid"
    ]

    def __init__(
        self,
        retrievers
    ):

        self.retrievers = retrievers

    def evaluate_strategy_actions(
        self,
        query,
        relevant_scores,
        query_type,
        current_strategy,
        top_k=5
    ):

        results = []

        for strategy in self.STRATEGIES:

            retriever = self.retrievers[
                strategy
            ]

            original_top_k = getattr(
                retriever,
                "top_k",
                5
            )

            try:

                retriever.top_k = top_k

                result = retriever.retrieve(
                    query
                )

            finally:

                retriever.top_k = original_top_k

            retrieved_ids = [
                chunk.chunk_id
                for chunk
                in result.retrieved_chunks
            ]

            recall = (
                RetrievalMetrics.recall_at_k(
                    retrieved_ids,
                    list(
                        relevant_scores.keys()
                    ),
                    top_k
                )
            )

            mrr = (
                self._mrr_at_k(
                    retrieved_ids,
                    list(
                        relevant_scores.keys()
                    ),
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

            if strategy == current_strategy:

                action = "keep"

            else:

                action = (
                    f"switch_to_{strategy}"
                )

            results.append({
                "query":
                    query,

                "query_type":
                    query_type,

                "current_strategy":
                    current_strategy,

                "candidate_strategy":
                    strategy,

                "action":
                    action,

                "top_k":
                    top_k,

                "recall":
                    recall,

                "mrr":
                    mrr,

                "ndcg":
                    ndcg,

                "utility":
                    ndcg
            })

        return results

    @staticmethod
    def _mrr_at_k(
        retrieved_ids,
        relevant_ids,
        k
    ):

        relevant = set(
            relevant_ids
        )

        for rank, document_id in enumerate(
            retrieved_ids[:k],
            start=1
        ):

            if document_id in relevant:

                return 1.0 / rank

        return 0.0