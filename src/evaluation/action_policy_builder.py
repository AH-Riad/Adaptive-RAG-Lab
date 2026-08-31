import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.action_evaluator import (
    ActionEvaluator
)

from src.evaluation.topk_action_evaluator import (
    TopKActionEvaluator
)

from src.planning.calibrated_policy import (
    CalibratedPolicy
)


class ActionPolicyBuilder:

    def __init__(
        self,
        output_path
    ):

        self.output_path = Path(
            output_path
        )

    def build(
        self,
        queries,
        qrels,
        query_types,
        retrievers
    ):

        strategy_evaluator = (
            ActionEvaluator(
                retrievers
            )
        )

        topk_evaluator = (
            TopKActionEvaluator(
                retrievers
            )
        )

        calibrated_policy = (
            CalibratedPolicy(
                path=(
                    "results/logs/"
                    "fiqa_dev_strategy_policy_v1.json"
                )
            )
        )

        strategy_groups = defaultdict(
            list
        )

        topk_groups = defaultdict(
            list
        )

        strategy_records = []
        topk_records = []

        for query_id, query in (
            queries.items()
        ):

            query_type = query_types[
                query_id
            ]

            relevant_scores = qrels.get(
                query_id,
                {}
            )

            current_strategy = (
                calibrated_policy
                .get_strategy(
                    query_type
                )
                .value
            )

            strategy_results = (
                strategy_evaluator
                .evaluate_strategy_actions(
                    query=query,
                    relevant_scores=(
                        relevant_scores
                    ),
                    query_type=query_type,
                    current_strategy=(
                        current_strategy
                    ),
                    top_k=5
                )
            )

            best_strategy = max(
                strategy_results,
                key=lambda item:
                    item["utility"]
            )

            strategy_state = (
                query_type,
                current_strategy
            )

            strategy_groups[
                strategy_state
            ].append(
                best_strategy
            )

            strategy_records.append({
                "query_id":
                    query_id,

                "query_type":
                    query_type,

                "current_strategy":
                    current_strategy,

                "best_action":
                    best_strategy[
                        "action"
                    ],

                "best_ndcg_at_5":
                    best_strategy[
                        "ndcg"
                    ],

                "candidates":
                    strategy_results
            })

            topk_results = (
                topk_evaluator.evaluate_query(
                    query=query,
                    relevant_scores=(
                        relevant_scores
                    ),
                    current_strategy=(
                        current_strategy
                    )
                )
            )

            best_topk = max(
                topk_results,
                key=lambda item:
                    item["utility"]
            )

            topk_state = (
                query_type,
                current_strategy
            )

            topk_groups[
                topk_state
            ].append(
                best_topk
            )

            topk_records.append({
                "query_id":
                    query_id,

                "query_type":
                    query_type,

                "current_strategy":
                    current_strategy,

                "best_action":
                    best_topk[
                        "action"
                    ],

                "best_top_k":
                    best_topk[
                        "top_k"
                    ],

                "best_ndcg":
                    best_topk[
                        "ndcg_at_k"
                    ],

                "candidates":
                    topk_results
            })

        strategy_policy = (
            self._aggregate(
                strategy_groups,
                action_key="action"
            )
        )

        topk_policy = (
            self._aggregate(
                topk_groups,
                action_key="action"
            )
        )

        artifact = {
            "dataset":
                "fiqa",

            "split":
                "dev",

            "version":
                "v1",

            "strategy_policy":
                strategy_policy,

            "topk_policy":
                topk_policy,

            "strategy_records":
                strategy_records,

            "topk_records":
                topk_records
        }

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                artifact,
                file,
                indent=2
            )

        return artifact

    @staticmethod
    def _aggregate(
        groups,
        action_key
    ):

        policy = {}

        for state, rows in (
            groups.items()
        ):

            action_groups = defaultdict(
                list
            )

            for row in rows:

                action_groups[
                    row[action_key]
                ].append(
                    row["utility"]
                )

            candidates = {}

            for action, values in (
                action_groups.items()
            ):

                candidates[action] = {
                    "count":
                        len(values),

                    "average_utility":
                        (
                            sum(values)
                            /
                            len(values)
                        )
                }

            best_action = max(
                candidates,
                key=lambda action:
                    candidates[action][
                        "average_utility"
                    ]
            )

            policy[
                str(state)
            ] = {
                "query_type":
                    state[0],

                "current_strategy":
                    state[1],

                "selected_action":
                    best_action,

                "candidates":
                    candidates
            }

        return policy