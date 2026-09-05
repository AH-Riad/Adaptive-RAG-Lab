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
        output_path,
        cost_weight: float = 0.10,
        minimum_gain: float = 0.03,
        minimum_query_support: int = 10
    ):

        self.output_path = Path(
            output_path
        )

        self.cost_weight = (
            cost_weight
        )

        self.minimum_gain = (
            minimum_gain
        )

        self.minimum_query_support = (
            minimum_query_support
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
                ),
                cost_weight=(
                    self.cost_weight
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

            current_retriever = (
                retrievers[
                    current_strategy
                ]
            )

            current_top_k = 5

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

            current_strategy_utility = max(
                item["utility"]
                for item in strategy_evaluations
                if item[
                    "candidate_strategy"
                ]
                == current_strategy
            )

            strategy_state = (
                query_type,
                current_strategy,
                confidence_bucket,
                current_top_k
            )

            for evaluation in (
                strategy_evaluations
            ):

                enriched = dict(
                    evaluation
                )

                enriched[
                    "query_id"
                ] = query_id

                enriched[
                    "confidence_bucket"
                ] = confidence_bucket

                enriched[
                    "current_utility"
                ] = current_strategy_utility

                strategy_groups[
                    strategy_state
                ].append(
                    enriched
                )

            best_strategy = max(
                strategy_evaluations,
                key=lambda item:
                    item["utility"]
            )

            strategy_gain = (
                best_strategy["utility"]
                -
                current_strategy_utility
            )

            if (
                best_strategy["candidate_strategy"]
                == current_strategy
                or
                strategy_gain
                < self.minimum_gain
            ):

                strategy_action = (
                    "keep"
                )

            else:

                strategy_action = (
                    best_strategy["action"]
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

                "selected_action":
                    strategy_action,

                "utility_gain":
                    strategy_gain
            })

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

                state = (
                    query_type,
                    current_strategy,
                    state_bucket,
                    state_top_k
                )

                topk_evaluations = (
                    topk_evaluator
                    .evaluate_query(
                        query=query,
                        relevant_scores=(
                            relevant_scores
                        ),
                        current_strategy=(
                            current_strategy
                        ),
                        current_top_k=(
                            state_top_k
                        )
                    )
                )

                current_item = next(
                    item
                    for item in (
                        topk_evaluations
                    )
                    if (
                        item["top_k"]
                        == state_top_k
                        and
                        item["action"]
                        == "keep"
                    )
                )

                current_utility = (
                    current_item[
                        "utility"
                    ]
                )

                for evaluation in (
                    topk_evaluations
                ):

                    enriched = dict(
                        evaluation
                    )

                    enriched[
                        "query_id"
                    ] = query_id

                    enriched[
                        "confidence_bucket"
                    ] = state_bucket

                    enriched[
                        "current_utility"
                    ] = current_utility

                    enriched[
                        "current_top_k"
                    ] = state_top_k

                    topk_groups[
                        state
                    ].append(
                        enriched
                    )

                best_topk = max(
                    topk_evaluations,
                    key=lambda item:
                        item["utility"]
                )

                topk_gain = (
                    best_topk["utility"]
                    -
                    current_utility
                )

                # --- GUIDE'S CORRECTED LOGIC APPLIED HERE ---
                if (
                    topk_gain < self.minimum_gain
                    or
                    best_topk["top_k"] == state_top_k
                ):

                    topk_action = "keep"

                else:

                    topk_action = (
                        best_topk["action"]
                    )
                # --------------------------------------------

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

                    "selected_action":
                        topk_action,

                    "best_top_k":
                        best_topk[
                            "top_k"
                        ],

                    "utility_gain":
                        topk_gain,

                    "incremental_recall":
                        best_topk.get("incremental_recall", 0.0) # Used .get() for safety
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
                "v5.1",

            "objective": {

                "strategy_utility":
                    "nDCG@5",

                "topk_utility":
                    (
                        "incremental Recall@K "
                        "- cost penalty"
                    ),

                "cost_weight":
                    self.cost_weight,

                "minimum_gain":
                    self.minimum_gain,

                "minimum_query_support":
                    self.minimum_query_support
            },

            "state_definition": [

                "query_type",
                "current_strategy",
                "confidence_bucket",
                "current_top_k"
            ],

            "topk_candidate_rule":
                (
                    "keep or expand to a "
                    "larger supported K"
                ),

            "supported_top_k":
                list(
                    self.SUPPORTED_TOP_K
                ),

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

    def _aggregate(
        self,
        groups
    ):

        policy = {}

        for state, rows in (
            groups.items()
        ):

            action_groups = (
                defaultdict(list)
            )

            query_ids = set()

            for row in rows:

                action_groups[
                    row["action"]
                ].append(
                    row["utility"]
                )

                if "query_id" in row:
                    query_ids.add(
                        str(
                            row[
                                "query_id"
                            ]
                        )
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

            query_count = len(
                query_ids
            )

            keep_utility = candidates.get(
                "keep",
                {
                    "average_utility": 0.0
                }
            )[
                "average_utility"
            ]

            eligible_actions = []

            for action, data in (
                candidates.items()
            ):

                if action == "keep":
                    continue

                gain = (
                    data["average_utility"]
                    -
                    keep_utility
                )

                if (
                    gain
                    >= self.minimum_gain
                    and
                    query_count
                    >= self.minimum_query_support
                ):

                    eligible_actions.append(
                        (
                            action,
                            data[
                                "average_utility"
                            ],
                            gain
                        )
                    )

            if eligible_actions:

                selected_action = max(
                    eligible_actions,
                    key=lambda item:
                        item[1]
                )[0]

            else:

                selected_action = (
                    "keep"
                )

            policy[
                str(state)
            ] = {

                "selected_action":
                    selected_action,

                "candidates":
                    candidates,

                "query_count":
                    query_count,

                "samples":
                    len(rows),

                "minimum_gain":
                    self.minimum_gain,

                "minimum_query_support":
                    self.minimum_query_support
            }

        return policy