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

from src.evaluation.evidence_calibrator import (
    EvidenceCalibrator
)

from src.core.adaptive_context import (
    AdaptiveContext
)

from src.planning.retrieval_plan import (
    RetrievalPlan
)

from src.planning.decision_types import (
    RetrievalStrategy
)


class ActionPolicyBuilder:

    SUPPORTED_TOP_K = (
        3,
        5,
        10,
        15
    )

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

        self.calibrator = (
            EvidenceCalibrator()
        )

        self.calibrator.load(
            "results/logs/"
            "fiqa_dev_evidence_calibrator_v1.json"
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

    def _build_evidence_state(
        self,
        query,
        query_type,
        retriever,
        strategy,
        top_k
    ):

        original_top_k = getattr(
            retriever,
            "top_k",
            5
        )

        try:

            retriever.top_k = top_k

            retrieval_result = (
                retriever.retrieve(
                    query
                )
            )

        finally:

            retriever.top_k = (
                original_top_k
            )

        context = AdaptiveContext(
            query=query
        )

        context.query_analysis = {
            "query_type":
                query_type
        }

        context.retrieval_plan = (
            RetrievalPlan(
                strategy=(
                    RetrievalStrategy(
                        strategy
                    )
                ),
                top_k=top_k,
                chunk_size=0,
                chunk_overlap=0
            )
        )

        context.retrieval_result = (
            retrieval_result
        )

        features = (
            self.feature_extractor.extract(
                context
            )
        )

        confidence = (
            self.calibrator.predict_probability(
                features
            )
        )

        return (
            retrieval_result,
            confidence,
            self.confidence_bucket(
                confidence
            )
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
                retrievers,
                candidate_top_k=(
                    self.SUPPORTED_TOP_K
                )
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

            current_top_k = 5

            current_retriever = (
                retrievers[
                    current_strategy
                ]
            )

            (
                _,
                current_confidence,
                confidence_bucket
            ) = self._build_evidence_state(
                query=query,
                query_type=query_type,
                retriever=current_retriever,
                strategy=current_strategy,
                top_k=current_top_k
            )

            strategy_evaluations = (
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
                    top_k=current_top_k
                )
            )

            for evaluation in (
                strategy_evaluations
            ):

                state = (
                    query_type,
                    current_strategy,
                    confidence_bucket,
                    current_top_k
                )

                strategy_groups[
                    state
                ].append(
                    evaluation
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

                "current_top_k":
                    current_top_k,

                "evidence_confidence":
                    current_confidence,

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

            # Build a genuine Top-K decision state
            # for each supported current Top-K.

            for state_top_k in (
                self.SUPPORTED_TOP_K
            ):

                (
                    _,
                    state_confidence,
                    state_bucket
                ) = self._build_evidence_state(
                    query=query,
                    query_type=query_type,
                    retriever=current_retriever,
                    strategy=current_strategy,
                    top_k=state_top_k
                )

                topk_evaluations = (
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

                state = (
                    query_type,
                    current_strategy,
                    state_bucket,
                    state_top_k
                )

                for evaluation in (
                    topk_evaluations
                ):

                    topk_groups[
                        state
                    ].append(
                        evaluation
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

                    "current_top_k":
                        state_top_k,

                    "evidence_confidence":
                        state_confidence,

                    "confidence_bucket":
                        state_bucket,

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
                "v3",

            "state_definition": [
                "query_type",
                "current_strategy",
                "confidence_bucket",
                "current_top_k"
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