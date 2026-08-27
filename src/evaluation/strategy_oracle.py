import json


class StrategyOracle:

    STRATEGIES = [
        "Dense",
        "BM25S",
        "Hybrid"
    ]

    def __init__(
        self,
        results_path: str
    ):

        with open(
            results_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.results = json.load(
                file
            )

    def _group_by_query(
        self
    ):

        grouped = {}

        for row in self.results:

            query_id = row[
                "query_id"
            ]

            grouped.setdefault(
                query_id,
                {}
            )

            grouped[
                query_id
            ][
                row["system"]
            ] = row

        return grouped

    def calculate(
        self
    ):

        grouped = self._group_by_query()

        total = len(
            grouped
        )

        correct = 0

        adaptive_attempts = 0

        successful_recoveries = 0

        failed_adaptations = 0

        initial_beats_final = 0

        final_beats_initial = 0

        strategy_transitions = 0

        oracle_regret = 0.0

        rows = []

        for query_id, systems in (
            grouped.items()
        ):

            if "D2RAG" not in systems:
                continue

            d2rag = systems[
                "D2RAG"
            ]

            strategy_scores = {
                strategy:
                    systems[strategy][
                        "ndcg_at_5"
                    ]
                for strategy
                in self.STRATEGIES
                if strategy in systems
            }

            if not strategy_scores:
                continue

            oracle_strategy = max(
                strategy_scores,
                key=strategy_scores.get
            )

            oracle_score = (
                strategy_scores[
                    oracle_strategy
                ]
            )

            initial_strategy = (
                d2rag[
                    "initial_strategy"
                ]
            )

            final_strategy = (
                d2rag[
                    "final_strategy"
                ]
            )

            initial_system = (
                self._map_strategy(
                    initial_strategy
                )
            )

            final_system = (
                self._map_strategy(
                    final_strategy
                )
            )

            initial_score = (
                strategy_scores.get(
                    initial_system,
                    0.0
                )
            )

            final_score = (
                d2rag[
                    "ndcg_at_5"
                ]
            )

            if initial_system == oracle_strategy:

                correct += 1

            changed = (
                initial_strategy
                != final_strategy
            )

            if changed:

                strategy_transitions += 1

            attempts = d2rag.get(
                "attempts",
                1
            )

            if attempts > 1:

                adaptive_attempts += 1

            if (
                changed
                and
                final_score > initial_score
            ):

                successful_recoveries += 1

            if (
                changed
                and
                final_score <= initial_score
            ):

                failed_adaptations += 1

            if initial_score > final_score:

                initial_beats_final += 1

            elif final_score > initial_score:

                final_beats_initial += 1

            oracle_regret += (
                oracle_score
                -
                final_score
            )

            rows.append({
                "query_id": query_id,
                "initial_strategy":
                    initial_strategy,
                "final_strategy":
                    final_strategy,
                "oracle_strategy":
                    oracle_strategy,
                "initial_ndcg":
                    initial_score,
                "final_ndcg":
                    final_score,
                "oracle_ndcg":
                    oracle_score,
                "adapted":
                    changed
            })

        evaluated = len(rows)

        return {
            "queries": evaluated,

            "strategy_selection_accuracy": (
                correct / evaluated
                if evaluated
                else 0.0
            ),

            "adaptation_rate": (
                adaptive_attempts / evaluated
                if evaluated
                else 0.0
            ),

            "strategy_transition_rate": (
                strategy_transitions / evaluated
                if evaluated
                else 0.0
            ),

            "successful_recovery_rate": (
                successful_recoveries
                /
                strategy_transitions
                if strategy_transitions
                else 0.0
            ),

            "failed_adaptation_rate": (
                failed_adaptations
                /
                strategy_transitions
                if strategy_transitions
                else 0.0
            ),

            "average_oracle_regret": (
                oracle_regret / evaluated
                if evaluated
                else 0.0
            ),

            "initial_better_count":
                initial_beats_final,

            "final_better_count":
                final_beats_initial,

            "query_details":
                rows
        }

    @staticmethod
    def _map_strategy(
        strategy
    ):

        mapping = {
            "dense": "Dense",
            "bm25": "BM25S",
            "hybrid": "Hybrid"
        }

        return mapping.get(
            strategy,
            strategy
        )