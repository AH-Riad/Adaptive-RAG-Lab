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

from src.planning.decision_engine import (
    DecisionEngine
)

from src.assessment.evidence_assessor import (
    EvidenceAssessor
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
        minimum_gain: float = 0.03
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

            current_strategy_utility = max(
                item["utility"]
                for item in strategy_evaluations
                if item["candidate_strategy"]
                == current_strategy
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

                enriched = dict(
                    evaluation
                )

                enriched[
                    "confidence_bucket"
                ] = confidence_bucket

                enriched[
                    "current_utility"
                ] = current_strategy_utility

                strategy_groups[
                    state
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

            if strategy_gain < (
                self.minimum_gain
            ):

                strategy_action = "keep"

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
                    best_strategy["action"],

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

                current_item = next(
                    item
                    for item
                    in topk_evaluations
                    if item[
                        "top_k"
                    ]
                    == state_top_k
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
                        "confidence_bucket"
                    ] = state_bucket

                    enriched[
                        "current_utility"
                    ] = current_utility

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

                if topk_gain < (
                    self.minimum_gain
                ):

                    topk_action = (
                        f"set_top_k_"
                        f"{state_top_k}"
                    )

                else:

                    topk_action = (
                        best_topk["action"]
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

                    "selected_action":
                        topk_action,

                    "best_top_k":
                        best_topk[
                            "top_k"
                        ],

                    "utility_gain":
                        topk_gain
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
                "v4",

            "objective": {
                "strategy_utility":
                    "nDCG@5",

                "topk_utility":
                    "nDCG@K - cost penalty",

                "cost_weight":
                    self.cost_weight,

                "minimum_gain":
                    self.minimum_gain
            },

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

            action_groups = defaultdict(
                list
            )

            for row in rows:

                action_groups[
                    row["action"]
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

            selected_action = max(
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
                    selected_action,

                "candidates":
                    candidates,

                "samples":
                    len(rows)
            }

        return policy

class FailureConditionedActionPolicyBuilder:
    """Build a post-retrieval action policy from rejected retrieval states only."""

    SUPPORTED_TOP_K = (3, 5, 8, 10, 15)

    STRATEGY_DIAGNOSES = {
        "retrieval_disagreement",
        "lexical_strategy_mismatch",
        "semantic_strategy_mismatch",
        "ambiguous_strategy_risk",
    }

    TOP_K_DIAGNOSES = {
        "coverage_gap",
        "ranking_uncertainty",
        "comparison_coverage_risk",
    }

    GENERIC_DIAGNOSES = {
        "no_evidence",
        "very_weak_evidence",
        "weak_evidence",
        "uncertain_failure",
    }

    def __init__(
        self,
        output_path,
        cost_weight: float = 0.10,
        minimum_gain: float = 0.03,
        minimum_query_support: int = 5,
        calibrator_path: str = "results/logs/fiqa_dev_evidence_calibrator_v1.json",
    ):
        self.output_path = Path(output_path)
        self.cost_weight = float(cost_weight)
        self.minimum_gain = float(minimum_gain)
        self.minimum_query_support = int(minimum_query_support)
        self.feature_extractor = EvidenceFeatureExtractor()
        self.decision_engine = DecisionEngine()
        self.evidence_assessor = EvidenceAssessor(
            calibrated_model_path=calibrator_path
        )

    @staticmethod
    def confidence_bucket(confidence):
        if confidence < 0.25:
            return "very_low"
        if confidence < 0.50:
            return "low"
        if confidence < 0.75:
            return "medium"
        return "high"

    @staticmethod
    def _set_top_k(retriever, top_k):
        objects = [
            retriever,
            getattr(retriever, "dense_retriever", None),
            getattr(retriever, "bm25_retriever", None),
            getattr(retriever, "hybrid_retriever", None),
        ]
        seen = set()
        for item in objects:
            if item is None:
                continue
            identifier = id(item)
            if identifier in seen:
                continue
            seen.add(identifier)
            if hasattr(item, "top_k"):
                item.top_k = top_k

    def _initial_plan(self, query, query_type):
        context = AdaptiveContext(query=query)
        context.query_analysis = {"query_type": query_type}
        context = self.decision_engine.run(context)
        return context.retrieval_plan

    def _build_evidence_state(
        self,
        query,
        query_type,
        retriever,
        strategy,
        top_k,
    ):
        original_values = {}
        for item in [
            retriever,
            getattr(retriever, "dense_retriever", None),
            getattr(retriever, "bm25_retriever", None),
            getattr(retriever, "hybrid_retriever", None),
        ]:
            if item is not None and hasattr(item, "top_k"):
                original_values[id(item)] = (item, item.top_k)

        try:
            self._set_top_k(retriever, top_k)
            retrieval_result = retriever.retrieve(query)
        finally:
            for item, value in original_values.values():
                item.top_k = value

        context = AdaptiveContext(query=query)
        context.query_analysis = {"query_type": query_type}
        context.retrieval_plan = RetrievalPlan(
            strategy=RetrievalStrategy(strategy),
            top_k=top_k,
            chunk_size=0,
            chunk_overlap=0,
        )
        context.retrieval_result = retrieval_result
        context = self.evidence_assessor.run(context)

        features = self.feature_extractor.extract(context)
        evidence = context.evidence_result
        confidence = max(0.0, min(1.0, float(evidence.confidence)))
        bucket = self.confidence_bucket(confidence)
        diagnosis, reason = self._diagnose(
            query_type=query_type,
            strategy=strategy,
            top_k=top_k,
            evidence=evidence,
            features=features,
            confidence=confidence,
        )

        return {
            "context": context,
            "features": features,
            "evidence": evidence,
            "confidence": confidence,
            "confidence_bucket": bucket,
            "diagnosis": diagnosis,
            "diagnosis_reason": reason,
        }

    @classmethod
    def _diagnose(
        cls,
        query_type,
        strategy,
        top_k,
        evidence,
        features,
        confidence,
    ):
        if evidence.retrieved_count == 0:
            return "no_evidence", "No evidence was retrieved."

        disagreement = features.get("dense_bm25_agreement")
        top1 = features.get("top1_score", 0.0)
        top1_top2_gap = features.get("top1_top2_gap", 0.0)
        score_coverage = getattr(evidence, "coverage", 0.0)

        if disagreement is not None and 0.0 < disagreement < 0.50:
            return (
                "retrieval_disagreement",
                "Dense and lexical retrieval signals disagree strongly.",
            )

        if score_coverage < 0.50 and evidence.retrieved_count >= 3:
            return (
                "coverage_gap",
                "Retrieved evidence has insufficient score-based coverage.",
            )

        if top1 >= 0.45 and top1_top2_gap < 0.05:
            return (
                "ranking_uncertainty",
                "Top-ranked evidence is weakly separated from the next result.",
            )

        if confidence < 0.25:
            return (
                "very_weak_evidence",
                "Calibrated evidence confidence is very low.",
            )

        if query_type == "lexical" and strategy == "dense":
            return (
                "lexical_strategy_mismatch",
                "The query is lexical but dense retrieval is currently being used.",
            )

        if query_type == "semantic" and strategy == "bm25":
            return (
                "semantic_strategy_mismatch",
                "The query is semantic but lexical retrieval is currently being used.",
            )

        if query_type == "ambiguous" and strategy == "dense":
            return (
                "ambiguous_strategy_risk",
                "The query is ambiguous and dense retrieval may benefit from lexical support.",
            )

        if query_type == "comparison" and strategy == "hybrid" and top_k < 10:
            return (
                "comparison_coverage_risk",
                "Comparison queries may require broader evidence coverage.",
            )

        if confidence < 0.50:
            return (
                "weak_evidence",
                "Evidence confidence is below the medium-confidence range.",
            )

        return (
            "uncertain_failure",
            "Evidence was rejected but no dominant failure pattern was detected.",
        )

    @staticmethod
    def _add_row(groups, state, action, utility, current_utility):
        groups[state].append({
            "action": action,
            "utility": float(utility),
            "utility_gain": float(utility - current_utility),
        })

    def _aggregate(self, groups):
        policy = {}

        for state, rows in groups.items():
            action_groups = defaultdict(list)
            for row in rows:
                action_groups[row["action"]].append(row)

            candidates = {}
            for action, action_rows in action_groups.items():
                gains = [row["utility_gain"] for row in action_rows]
                utilities = [row["utility"] for row in action_rows]
                candidates[action] = {
                    "count": len(action_rows),
                    "average_gain": sum(gains) / len(gains),
                    "average_utility": sum(utilities) / len(utilities),
                }

            if "keep" not in candidates:
                candidates["keep"] = {
                    "count": len(rows),
                    "average_gain": 0.0,
                    "average_utility": 0.0,
                }

            keep = candidates["keep"]

            eligible = {
                action: record
                for action, record in candidates.items()
                if action == "keep"
                or record["count"] >= self.minimum_query_support
            }

            non_keep = {
                action: record
                for action, record in eligible.items()
                if action != "keep"
            }

            best_action = "keep"
            if non_keep:
                candidate_action = max(
                    non_keep,
                    key=lambda action: (
                        non_keep[action]["average_gain"],
                        non_keep[action]["average_utility"],
                        action,
                    ),
                )
                candidate = non_keep[candidate_action]
                keep_gain = keep.get("average_gain", 0.0)
                if (
                    candidate["average_gain"] - keep_gain
                    >= self.minimum_gain
                ):
                    best_action = candidate_action

            policy[str(state)] = {
                "selected_action": best_action,
                "candidates": eligible,
                "samples": len(rows),
                "selection_rule": {
                    "minimum_gain": self.minimum_gain,
                    "minimum_query_support": self.minimum_query_support,
                    "objective": "average cost-adjusted utility gain over KEEP",
                },
            }

        return policy

    def build(self, queries, qrels, query_types, retrievers):
        strategy_evaluator = ActionEvaluator(retrievers)
        topk_evaluator = TopKActionEvaluator(
            retrievers,
            candidate_top_k=self.SUPPORTED_TOP_K,
            cost_weight=self.cost_weight,
        )

        strategy_groups = defaultdict(list)
        topk_groups = defaultdict(list)
        combined_groups = defaultdict(list)
        state_records = []
        rejected_states = 0
        accepted_states = 0

        for query_id, query in queries.items():
            query_type = query_types[query_id]
            relevant_scores = qrels.get(query_id, {})
            initial_plan = self._initial_plan(query, query_type)
            current_strategy = initial_plan.strategy.value

            for state_top_k in self.SUPPORTED_TOP_K:
                retriever = retrievers[current_strategy]
                state = self._build_evidence_state(
                    query=query,
                    query_type=query_type,
                    retriever=retriever,
                    strategy=current_strategy,
                    top_k=state_top_k,
                )

                if state["evidence"].accepted:
                    accepted_states += 1
                    continue

                rejected_states += 1
                diagnosis = state["diagnosis"]
                confidence_bucket = state["confidence_bucket"]
                policy_state = (
                    query_type,
                    current_strategy,
                    confidence_bucket,
                    diagnosis,
                    state_top_k,
                )

                strategy_evaluations = strategy_evaluator.evaluate_strategy_actions(
                    query=query,
                    relevant_scores=relevant_scores,
                    query_type=query_type,
                    current_strategy=current_strategy,
                    top_k=state_top_k,
                )
                current_strategy_utility = max(
                    item["utility"]
                    for item in strategy_evaluations
                    if item["candidate_strategy"] == current_strategy
                )

                for evaluation in strategy_evaluations:
                    self._add_row(
                        strategy_groups,
                        policy_state,
                        evaluation["action"],
                        evaluation["utility"],
                        current_strategy_utility,
                    )
                    if evaluation["action"] != "keep":
                        self._add_row(
                            combined_groups,
                            policy_state,
                            evaluation["action"],
                            evaluation["utility"],
                            current_strategy_utility,
                        )

                topk_evaluations = topk_evaluator.evaluate_query(
                    query=query,
                    relevant_scores=relevant_scores,
                    current_strategy=current_strategy,
                    current_top_k=state_top_k,
                )
                current_topk_item = next(
                    item
                    for item in topk_evaluations
                    if item["top_k"] == state_top_k
                )
                current_topk_utility = current_topk_item["utility"]

                for evaluation in topk_evaluations:
                    self._add_row(
                        topk_groups,
                        policy_state,
                        evaluation["action"],
                        evaluation["utility"],
                        current_topk_utility,
                    )
                    if evaluation["action"] != "keep":
                        self._add_row(
                            combined_groups,
                            policy_state,
                            evaluation["action"],
                            evaluation["utility"],
                            current_topk_utility,
                        )

                keep_utility = max(
                    current_strategy_utility,
                    current_topk_utility,
                )
                self._add_row(
                    combined_groups,
                    policy_state,
                    "keep",
                    keep_utility,
                    keep_utility,
                )

                state_records.append({
                    "query_id": query_id,
                    "query_type": query_type,
                    "current_strategy": current_strategy,
                    "current_top_k": state_top_k,
                    "evidence_confidence": state["confidence"],
                    "confidence_bucket": confidence_bucket,
                    "diagnosis": diagnosis,
                    "diagnosis_reason": state["diagnosis_reason"],
                    "accepted": False,
                })

        artifact = {
            "dataset": "fiqa",
            "split": "dev",
            "version": "v6",
            "policy_type": "failure_conditioned_dual_action_policy",
            "training_filter": "evidence_rejected_only",
            "objective": {
                "strategy_utility": "nDCG@K with evaluator cost model",
                "topk_utility": "nDCG@K - cost penalty",
                "selection_objective": "average cost-adjusted utility gain over KEEP",
                "cost_weight": self.cost_weight,
                "minimum_gain": self.minimum_gain,
                "minimum_query_support": self.minimum_query_support,
            },
            "state_definition": [
                "query_type",
                "current_strategy",
                "confidence_bucket",
                "diagnosis",
                "current_top_k",
            ],
            "supported_top_k": list(self.SUPPORTED_TOP_K),
            "strategy_policy": self._aggregate(strategy_groups),
            "topk_policy": self._aggregate(topk_groups),
            "combined_policy": self._aggregate(combined_groups),
            "state_records": state_records,
            "training_summary": {
                "accepted_states_skipped": accepted_states,
                "rejected_states_used": rejected_states,
                "policy_states_strategy": len(strategy_groups),
                "policy_states_topk": len(topk_groups),
                "policy_states_combined": len(combined_groups),
            },
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2)

        return artifact
