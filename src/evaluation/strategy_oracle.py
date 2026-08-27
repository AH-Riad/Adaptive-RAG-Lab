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

    def _group_by_query(self):

        grouped = {}

        for row in self.results:

            query_id = row.get(
                "query_id"
            )

            if query_id is None:
                continue

            grouped.setdefault(
                query_id,
                {}
            )

            grouped[
                query_id
            ][
                row.get("system")
            ] = row

        return grouped

    def calculate(self):

        grouped = self._group_by_query()

        rows = []

        eligible_queries = 0
        selection_correct = 0

        adapted_queries = 0
        successful_recoveries = 0
        failed_adaptations = 0

        strategy_transitions = 0

        oracle_regret = 0.0

        for query_id, systems in (
            grouped.items()
        ):

            d2rag = systems.get(
                "D2RAG"
            )

            if d2rag is None:
                continue

            available_scores = {}

            for strategy in self.STRATEGIES:

                row = systems.get(
                    strategy
                )

                if row is None:
                    continue

                score = row.get(
                    "ndcg_at_5"
                )

                if score is None:
                    continue

                available_scores[
                    strategy
                ] = float(score)

            if not available_scores:
                continue

            initial_strategy = (
                self._normalize_strategy(
                    d2rag.get(
                        "initial_strategy"
                    )
                )
            )

            final_strategy = (
                self._normalize_strategy(
                    d2rag.get(
                        "final_strategy"
                    )
                )
            )

            final_score = self._safe_float(
                d2rag.get(
                    "ndcg_at_5"
                )
            )

            initial_score = (
                available_scores.get(
                    initial_strategy
                )
                if initial_strategy
                else None
            )

            best_score = max(
                available_scores.values()
            )

            best_strategies = [
                strategy
                for strategy, score
                in available_scores.items()
                if abs(
                    score - best_score
                ) < 1e-12
            ]

            # No meaningful oracle winner if
            # all available strategies score zero.

            if best_score <= 0.0:

                oracle_strategy = None

                oracle_score = 0.0

                strategy_correct = None

            else:

                oracle_strategy = (
                    best_strategies[0]
                )

                oracle_score = best_score

                if initial_strategy is None:

                    strategy_correct = False

                else:

                    strategy_correct = (
                        initial_strategy
                        in best_strategies
                    )

                    if strategy_correct:

                        selection_correct += 1

                eligible_queries += 1

            changed = (
                initial_strategy is not None
                and
                final_strategy is not None
                and
                initial_strategy
                != final_strategy
            )

            if changed:

                strategy_transitions += 1

            attempts = d2rag.get(
                "attempts",
                1
            )

            if attempts is None:
                attempts = 1

            if attempts > 1:

                adapted_queries += 1

            recovery = False
            failed_adaptation = False

            if (
                changed
                and
                initial_score is not None
                and
                final_score is not None
            ):

                if final_score > initial_score:

                    recovery = True

                    successful_recoveries += 1

                else:

                    failed_adaptation = True

                    failed_adaptations += 1

            regret = None

            if final_score is not None:

                regret = (
                    oracle_score
                    -
                    final_score
                )

                if oracle_strategy is not None:

                    oracle_regret += regret

            rows.append({
                "query_id": query_id,

                "initial_strategy":
                    d2rag.get(
                        "initial_strategy"
                    ),

                "final_strategy":
                    d2rag.get(
                        "final_strategy"
                    ),

                "oracle_strategy":
                    oracle_strategy,

                "initial_ndcg":
                    initial_score,

                "final_ndcg":
                    final_score,

                "oracle_ndcg":
                    oracle_score,

                "adapted":
                    changed,

                "strategy_correct":
                    strategy_correct,

                "successful_recovery":
                    recovery,

                "failed_adaptation":
                    failed_adaptation,

                "oracle_regret":
                    regret
            })

        transition_denominator = (
            strategy_transitions
        )

        return {
            "queries": len(rows),

            "oracle_eligible_queries":
                eligible_queries,

            "zero_oracle_queries": (
                len(rows)
                -
                eligible_queries
            ),

            "strategy_selection_accuracy": (
                selection_correct
                /
                eligible_queries
                if eligible_queries
                else 0.0
            ),

            "adaptation_rate": (
                adapted_queries
                /
                len(rows)
                if rows
                else 0.0
            ),

            "strategy_transition_rate": (
                strategy_transitions
                /
                len(rows)
                if rows
                else 0.0
            ),

            "successful_recovery_rate": (
                successful_recoveries
                /
                transition_denominator
                if transition_denominator
                else 0.0
            ),

            "failed_adaptation_rate": (
                failed_adaptations
                /
                transition_denominator
                if transition_denominator
                else 0.0
            ),

            "average_oracle_regret": (
                oracle_regret
                /
                eligible_queries
                if eligible_queries
                else 0.0
            ),

            "query_details":
                rows
        }

    @staticmethod
    def _normalize_strategy(
        strategy
    ):

        if strategy is None:
            return None

        mapping = {
            "dense": "Dense",
            "bm25": "BM25S",
            "bm25s": "BM25S",
            "hybrid": "Hybrid"
        }

        return mapping.get(
            str(strategy).lower()
        )

    @staticmethod
    def _safe_float(
        value
    ):

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError
        ):

            return None