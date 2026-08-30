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

    def evaluate_query(
        self,
        query,
        relevant_scores,
        query_type,
        current_strategy,
        current_top_k=5
    ):

        evaluations = []

        relevant_ids = list(
            relevant_scores.keys()
        )

        # Evaluate keeping the current strategy
        current_retriever = self.retrievers[
            current_strategy
        ]

        current_result = (
            current_retriever.retrieve(
                query
            )
        )

        current_ids = [
            chunk.chunk_id
            for chunk
            in current_result.retrieved_chunks
        ]

        current_ndcg = (
            RetrievalMetrics.ndcg_at_k(
                current_ids,
                relevant_scores,
                5
            )
        )

        evaluations.append({
            "query":
                query,

            "query_type":
                query_type,

            "current_strategy":
                current_strategy,

            "current_top_k":
                current_top_k,

            "action":
                "keep",

            "target_strategy":
                current_strategy,

            "target_top_k":
                current_top_k,

            "ndcg_at_5":
                current_ndcg
        })

        # Evaluate strategy switches
        for strategy in self.STRATEGIES:

            if strategy == current_strategy:
                continue

            retriever = self.retrievers[
                strategy
            ]

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

            evaluations.append({
                "query":
                    query,

                "query_type":
                    query_type,

                "current_strategy":
                    current_strategy,

                "current_top_k":
                    current_top_k,

                "action":
                    f"switch_to_{strategy}",

                "target_strategy":
                    strategy,

                "target_top_k":
                    5,

                "ndcg_at_5":
                    ndcg
            })

        return evaluations