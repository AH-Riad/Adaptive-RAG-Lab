import csv
import json
from pathlib import Path

from src.core.adaptive_context import AdaptiveContext

from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.benchmark_result import (
    BenchmarkResult
)


class BenchmarkRunner:

    def __init__(
        self,
        ground_truth,
        output_dir: str = "results"
    ):

        self.ground_truth = ground_truth

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

    def _calculate_metrics(
        self,
        retrieved_ids,
        ground_truth_item
    ):

        precision = (
            RetrievalMetrics.precision_at_k(
                retrieved_ids,
                ground_truth_item.relevant_chunks,
                5
            )
        )

        recall = (
            RetrievalMetrics.recall_at_k(
                retrieved_ids,
                ground_truth_item.relevant_chunks,
                5
            )
        )

        mrr = (
            RetrievalMetrics.reciprocal_rank(
                retrieved_ids,
                ground_truth_item.relevant_chunks
            )
        )

        ndcg = (
            RetrievalMetrics.ndcg_at_k(
                retrieved_ids,
                ground_truth_item.relevance_scores,
                5
            )
        )

        return (
            precision,
            recall,
            mrr,
            ndcg
        )

    def run_baselines(
        self,
        retrievers: dict
    ):

        results = []

        for system_name, retriever in (
            retrievers.items()
        ):

            for item in (
                self.ground_truth.get_all()
            ):

                retrieval_result = (
                    retriever.retrieve(
                        item.query
                    )
                )

                retrieved_ids = [
                    chunk.chunk_id
                    for chunk
                    in retrieval_result.retrieved_chunks
                ]

                (
                    precision,
                    recall,
                    mrr,
                    ndcg
                ) = self._calculate_metrics(
                    retrieved_ids,
                    item
                )

                system_strategy = (
                    system_name.lower()
                )

                results.append(
                    BenchmarkResult(
                        system=system_name,

                        query=item.query,

                        query_type=item.query_type,

                        precision_at_5=precision,

                        recall_at_5=recall,

                        mrr=mrr,

                        ndcg_at_5=ndcg,

                        attempts=1,

                        initial_strategy=(
                            system_strategy
                        ),

                        final_strategy=(
                            system_strategy
                        ),

                        initial_top_k=5,

                        final_top_k=5,

                        planner_confidence=1.0,

                        evidence_confidence=0.0,

                        accepted=True
                    )
                )

        return results

    def run_d2rag(
        self,
        engine
    ):

        results = []

        for item in (
            self.ground_truth.get_all()
        ):

            context = AdaptiveContext(
                query=item.query
            )

            context = engine.run(
                context
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk
                in context.retrieval_result.retrieved_chunks
            ]

            (
                precision,
                recall,
                mrr,
                ndcg
            ) = self._calculate_metrics(
                retrieved_ids,
                item
            )

            report = (
                context.decision_report
            )

            transitions = [
                (
                    f"{transition['old_strategy']}"
                    f"->{transition['new_strategy']}"
                )
                for transition
                in report.get(
                    "strategy_transitions",
                    []
                )
            ]

            results.append(
                BenchmarkResult(
                    system="D2RAG",

                    query=item.query,

                    query_type=item.query_type,

                    precision_at_5=precision,

                    recall_at_5=recall,

                    mrr=mrr,

                    ndcg_at_5=ndcg,

                    attempts=report.get(
                        "retrieval_attempts",
                        1
                    ),

                    initial_strategy=report.get(
                        "initial_strategy",
                        ""
                    ),

                    final_strategy=report.get(
                        "final_strategy",
                        ""
                    ),

                    initial_top_k=report.get(
                        "initial_top_k",
                        0
                    ),

                    final_top_k=report.get(
                        "final_top_k",
                        0
                    ),

                    planner_confidence=report.get(
                        "initial_planner_confidence",
                        0.0
                    ),

                    evidence_confidence=report.get(
                        "final_evidence_confidence",
                        0.0
                    ),

                    accepted=(
                        context.evidence_result.accepted
                    ),

                    strategy_changes=transitions
                )
            )

        return results

    def save_results(
        self,
        results
    ):

        jsonl_path = (
            self.output_dir
            / "logs"
            / "benchmark_results.jsonl"
        )

        csv_path = (
            self.output_dir
            / "tables"
            / "benchmark_results.csv"
        )

        with jsonl_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            for result in results:

                file.write(
                    json.dumps(
                        result.to_dict()
                    )
                    + "\n"
                )

        if not results:
            return (
                jsonl_path,
                csv_path
            )

        fieldnames = list(
            results[0].to_dict().keys()
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for result in results:

                writer.writerow(
                    result.to_dict()
                )

        return (
            jsonl_path,
            csv_path
        )

    def aggregate(
        self,
        results
    ):

        systems = {}

        for result in results:

            systems.setdefault(
                result.system,
                []
            ).append(result)

        summary = {}

        for system, system_results in (
            systems.items()
        ):

            count = len(
                system_results
            )

            summary[system] = {
                "precision_at_5": (
                    sum(
                        r.precision_at_5
                        for r
                        in system_results
                    )
                    / count
                ),

                "recall_at_5": (
                    sum(
                        r.recall_at_5
                        for r
                        in system_results
                    )
                    / count
                ),

                "mrr": (
                    sum(
                        r.mrr
                        for r
                        in system_results
                    )
                    / count
                ),

                "ndcg_at_5": (
                    sum(
                        r.ndcg_at_5
                        for r
                        in system_results
                    )
                    / count
                )
            }

        return summary