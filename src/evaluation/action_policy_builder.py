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

from src.assessment.evidence_features import (
    EvidenceFeatureExtractor
)


class ActionPolicyBuilder:

    def __init__(
        self,
        output_path
    ):

        self.output_path = Path(
            output_path
        )

        self.feature_extractor = (
            EvidenceFeatureExtractor()
        )

    @staticmethod
    def confidence_bucket(
        confidence
    ):

        if confidence < 0.25:

            return "very_low"

        if confidence < 0.50:

            return "low"

        if confidence < 0.75:

            return "medium"

        return "high"

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

            strategy_evaluations = (
                strategy_evaluator
                .evaluate_strategy_actions(
                    query=query,
                    relevant_scores=relevant_scores,
                    query_type=query_type,
                    current_strategy=current_strategy,
                    top_k=5
                )
            )

            current_result = next(
                item
                for item in strategy_evaluations
                if item[
                    "candidate_strategy"
                ] == current_strategy
            )

            current_confidence = (
                current_result[
                    "utility"
                ]
            )

            confidence_bucket = (
                self.confidence_bucket(
                    current_confidence
                )
            )

            for item in strategy_evaluations:

                state = (
                    query_type,
                    current_strategy,
                    confidence_bucket,
                    5
                )

                strategy_groups[
                    state
                ].append(
                    item
                )

            best_strategy = max(
                strategy_evaluations,
                key=lambda item:
                    item["utility"]
            )

            strategy_records.append({
                "query_id":
                    query_id,

                "query_type":
                    query_type,

                "current_strategy":
                    current_strategy,

                "confidence_bucket":
                    confidence_bucket,

                "best_action":
                    best_strategy[
                        "action"
                    ],

                "best_ndcg_at_5":
                    best_strategy[
                        "ndcg"
                    ]
            })

            topk_evaluations = (
                topk_evaluator.evaluate_query(
                    query=query,
                    relevant_scores=relevant_scores,
                    current_strategy=current_strategy
                )
            )

            for item in topk_evaluations:

                state = (
                    query_type,
                    current_strategy,
                    confidence_bucket,
                    item[
                        "top_k"
                    ]
                )

                topk_groups[
                    state
                ].append(
                    item
                )

            best_topk = max(
                topk_evaluations,
                key=lambda item:
                    item["utility"]
            )

            topk_records.append({
                "query_id":
                    query_id,

                "query_type":
                    query_type,

                "current_strategy":
                    current_strategy,

                "confidence_bucket":
                    confidence_bucket,

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
                    ]
            })

        strategy_policy = (
            self._aggregate(
                strategy_groups
            )
        )

        topk_policy = (
            self._aggregate(
                topk_groups
            )
        )

        artifact = {
            "dataset":
                "fiqa",

            "split":
                "dev",

            "version":
                "v2",

            "state_definition": [
                "query_type",
                "current_strategy",
                "confidence_bucket",
                "top_k"
            ],

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
        groups
    ):

        policy = {}

        for state, rows in (
            groups.items()
        ):

            grouped_actions = defaultdict(
                list
            )

            for row in rows:

                grouped_actions[
                    row["action"]
                ].append(
                    row["utility"]
                )

            candidates = {}

            for action, values in (
                grouped_actions.items()
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
                "selected_action":
                    best_action,

                "candidates":
                    candidates,

                "samples":
                    len(rows)
            }

        return policy