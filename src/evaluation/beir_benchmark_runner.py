import json
from pathlib import Path

from src.core.adaptive_context import AdaptiveContext
from src.evaluation.metrics import RetrievalMetrics


class BEIRBenchmarkRunner:

    def __init__(
        self,
        queries,
        qrels,
        output_dir="results"
    ):

        self.queries = queries
        self.qrels = qrels

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        (
            self.output_dir / "logs"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            self.output_dir / "tables"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    def _metrics(
        self,
        retrieved_ids,
        relevant_scores
    ):

        relevant_ids = list(
            relevant_scores.keys()
        )

        precision = (
            RetrievalMetrics.precision_at_k(
                retrieved_ids,
                relevant_ids,
                5
            )
        )

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

        return {
            "precision_at_5": precision,
            "recall_at_5": recall,
            "mrr_at_5": mrr,
            "ndcg_at_5": ndcg
        }

    def evaluate_retriever(
        self,
        name,
        retriever,
        query_ids=None
    ):

        if query_ids is None:

            query_ids = list(
                self.queries.keys()
            )

        results = []

        for query_id in query_ids:

            query = self.queries[
                query_id
            ]

            result = retriever.retrieve(
                query
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk
                in result.retrieved_chunks
            ]

            metrics = self._metrics(
                retrieved_ids,
                self.qrels.get(
                    query_id,
                    {}
                )
            )

            results.append({
                "system": name,
                "query_id": query_id,
                "query": query,
                **metrics,

                "initial_strategy":
                    name.lower(),

                "final_strategy":
                    name.lower(),

                "initial_top_k":
                    5,

                "final_top_k":
                    5,

                "planner_confidence":
                    1.0,

                "evidence_confidence":
                    None,

                "accepted":
                    None,

                "attempts":
                    1,

                "status":
                    "fixed_baseline",

                "strategy_changes":
                    0
            })

        return results

    def evaluate_d2rag(
        self,
        engine,
        query_ids=None
    ):

        if query_ids is None:

            query_ids = list(
                self.queries.keys()
            )

        results = []

        for query_id in query_ids:

            query = self.queries[
                query_id
            ]

            context = AdaptiveContext(
                query=query
            )

            context = engine.run(
                context
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk
                in context.retrieval_result.retrieved_chunks
            ]

            metrics = self._metrics(
                retrieved_ids,
                self.qrels.get(
                    query_id,
                    {}
                )
            )

            report = (
                context.decision_report
            )

            evidence = (
                context.evidence_result
            )

            transitions = (
                report.get(
                    "strategy_transitions",
                    []
                )
            )

            results.append({
                "system": "D2RAG",

                "query_id": query_id,

                "query": query,

                **metrics,

                "initial_strategy":
                    report.get(
                        "initial_strategy",
                        ""
                    ),

                "final_strategy":
                    report.get(
                        "final_strategy",
                        ""
                    ),

                "initial_top_k":
                    report.get(
                        "initial_top_k",
                        0
                    ),

                "final_top_k":
                    report.get(
                        "final_top_k",
                        0
                    ),

                "planner_confidence":
                    report.get(
                        "initial_planner_confidence",
                        0.0
                    ),

                "evidence_confidence":
                    report.get(
                        "final_evidence_confidence",
                        0.0
                    ),

                "accepted":
                    evidence.accepted
                    if evidence is not None
                    else False,

                "attempts":
                    report.get(
                        "retrieval_attempts",
                        1
                    ),

                "status":
                    report.get(
                        "adaptive_retrieval_status",
                        "unknown"
                    ),

                "strategy_changes":
                    len(transitions)
            })

        return results

    def summarize(
        self,
        results
    ):

        systems = {}

        for result in results:

            systems.setdefault(
                result["system"],
                []
            ).append(result)

        summary = {}

        for system, rows in (
            systems.items()
        ):

            count = len(rows)

            summary[system] = {
                "queries":
                    count,

                "precision_at_5": (
                    sum(
                        row[
                            "precision_at_5"
                        ]
                        for row in rows
                    )
                    / count
                ),

                "recall_at_5": (
                    sum(
                        row[
                            "recall_at_5"
                        ]
                        for row in rows
                    )
                    / count
                ),

                "mrr_at_5": (
                    sum(
                        row[
                            "mrr_at_5"
                        ]
                        for row in rows
                    )
                    / count
                ),

                "ndcg_at_5": (
                    sum(
                        row[
                            "ndcg_at_5"
                        ]
                        for row in rows
                    )
                    / count
                )
            }

        return summary

    def save(
        self,
        results,
        filename="fiqa_benchmark_results.json"
    ):

        path = (
            self.output_dir
            / "logs"
            / filename
        )

        with path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=2
            )

        return path