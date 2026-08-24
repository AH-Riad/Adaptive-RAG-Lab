from collections import defaultdict

from src.evaluation.metrics import RetrievalMetrics


class StrategyCalibrator:

    def __init__(
        self,
        query_analyzer,
        retrievers
    ):
        self.query_analyzer = query_analyzer
        self.retrievers = retrievers

    def calibrate(
        self,
        queries,
        qrels
    ):

        grouped_results = defaultdict(
            lambda: defaultdict(list)
        )

        for query_id, query in queries.items():

            analysis = (
                self.query_analyzer.analyze(
                    query
                )
            )

            query_type = analysis[
                "query_type"
            ]

            relevant_scores = qrels.get(
                query_id,
                {}
            )

            relevant_ids = list(
                relevant_scores.keys()
            )

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

                recall = (
                    RetrievalMetrics.recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        5
                    )
                )

                mrr = (
                    RetrievalMetrics.reciprocal_rank_at_k(
                        retrieved_ids,
                        relevant_ids,
                        5
                    )
                )

                ndcg = (
                    RetrievalMetrics.ndcg_at_k(
                        retrieved_ids,
                        relevant_scores,
                        5
                    )
                )

                grouped_results[
                    query_type
                ][strategy].append({
                    "recall_at_5": recall,
                    "mrr_at_5": mrr,
                    "ndcg_at_5": ndcg
                })

        policy = {}

        report = {}

        for query_type, strategies in (
            grouped_results.items()
        ):

            strategy_scores = {}

            for strategy, rows in (
                strategies.items()
            ):

                count = len(rows)

                average_recall = (
                    sum(
                        row["recall_at_5"]
                        for row in rows
                    )
                    / count
                )

                average_mrr = (
                    sum(
                        row["mrr_at_5"]
                        for row in rows
                    )
                    / count
                )

                average_ndcg = (
                    sum(
                        row["ndcg_at_5"]
                        for row in rows
                    )
                    / count
                )

                combined_score = (
                    0.30 * average_recall
                    +
                    0.30 * average_mrr
                    +
                    0.40 * average_ndcg
                )

                strategy_scores[
                    strategy
                ] = {
                    "recall_at_5":
                        average_recall,

                    "mrr_at_5":
                        average_mrr,

                    "ndcg_at_5":
                        average_ndcg,

                    "combined_score":
                        combined_score
                }

            best_strategy = max(
                strategy_scores,
                key=lambda strategy:
                    strategy_scores[
                        strategy
                    ]["combined_score"]
            )

            policy[
                query_type
            ] = best_strategy

            report[
                query_type
            ] = {
                "selected_strategy":
                    best_strategy,

                "strategies":
                    strategy_scores
            }

        return {
            "policy": policy,
            "report": report
        }