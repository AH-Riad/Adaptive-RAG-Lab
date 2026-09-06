from src.core.component import Component
from src.adaptation.feedback_decision import FeedbackDecision
from src.adaptation.calibrated_feedback_policy import CalibratedFeedbackPolicy
from src.assessment.evidence_features import EvidenceFeatureExtractor


class FeedbackController(Component):

    SUPPORTED_TOP_K = (3, 5, 10, 15)

    ACTION_COSTS = {
        "keep": 0.0,
        "increase_top_k": 0.20,
        "switch_strategy": 0.40
    }

    def __init__(
        self,
        policy_path: str,
        minimum_samples: int = 5,
        minimum_improvement: float = 0.03
    ):
        self.policy = CalibratedFeedbackPolicy(
            policy_path=policy_path,
            minimum_samples=minimum_samples
        )
        self.feature_extractor = EvidenceFeatureExtractor()
        self.minimum_improvement = minimum_improvement

    @staticmethod
    def _confidence_bucket(confidence):
        if confidence < 0.25:
            return "very_low"

        if confidence < 0.50:
            return "low"

        if confidence < 0.75:
            return "medium"

        return "high"

    def _history(self, context):
        history = getattr(
            context,
            "feedback_history",
            None
        )

        if history is None:
            history = []
            context.feedback_history = history

        return history

    def _previous_actions(self, context):
        actions = set()

        for item in self._history(context):
            action = item.get("action")

            if action:
                actions.add(action)

        return actions

    def _diagnose(
        self,
        context,
        features,
        confidence
    ):
        plan = context.retrieval_plan
        evidence = context.evidence_result

        query_analysis = context.query_analysis or {}

        query_type = query_analysis.get(
            "query_type",
            "ambiguous"
        )

        strategy = plan.strategy.value

        top_k = plan.top_k

        if evidence.retrieved_count == 0:
            return (
                "no_evidence",
                "No evidence was retrieved."
            )

        coverage = evidence.coverage

        disagreement = features.get(
            "dense_bm25_agreement",
            0.0
        )

        top1 = features.get(
            "top1_score",
            0.0
        )

        top1_top2_gap = features.get(
            "top1_top2_gap",
            0.0
        )

        if query_type == "lexical" and strategy == "dense":
            return (
                "lexical_strategy_mismatch",
                "The query is lexical but dense retrieval is currently being used."
            )

        if query_type == "semantic" and strategy == "bm25":
            return (
                "semantic_strategy_mismatch",
                "The query is semantic but lexical retrieval is currently being used."
            )

        if query_type == "ambiguous" and strategy == "dense":
            return (
                "ambiguous_strategy_risk",
                "The query is ambiguous and dense retrieval may benefit from lexical support."
            )

        if query_type == "comparison" and strategy == "dense":
            return (
                "comparison_strategy_mismatch",
                "The comparison query may benefit from hybrid retrieval rather than dense-only retrieval."
            )

        if query_type == "comparison" and strategy == "bm25":
            return (
                "comparison_strategy_mismatch",
                "The comparison query may benefit from hybrid retrieval rather than lexical-only retrieval."
            )

        if disagreement > 0.0 and disagreement < 0.50:
            return (
                "retrieval_disagreement",
                "Dense and lexical retrieval signals disagree strongly."
            )

        if coverage < 0.50 and evidence.retrieved_count >= 3:
            return (
                "coverage_gap",
                "Retrieved evidence has insufficient score-based coverage."
            )

        if top1 >= 0.45 and top1_top2_gap < 0.05:
            return (
                "ranking_uncertainty",
                "Top-ranked evidence is weakly separated from the next result."
            )

        if confidence < 0.25:
            return (
                "very_weak_evidence",
                "Calibrated evidence confidence is very low."
            )

        if query_type == "comparison" and strategy == "hybrid" and top_k < 10:
            return (
                "comparison_coverage_risk",
                "Comparison queries may require broader evidence coverage."
            )

        if confidence < 0.50:
            return (
                "weak_evidence",
                "Evidence confidence is below the medium-confidence range."
            )

        return (
            "uncertain_failure",
            "Evidence was rejected but no dominant failure pattern was detected."
        )

    def _next_larger_top_k(self, current_top_k):
        for top_k in self.SUPPORTED_TOP_K:
            if top_k > current_top_k:
                return top_k

        return None

    def _strategy_action(
        self,
        query_type,
        current_strategy
    ):
        if query_type == "lexical":

            if current_strategy == "dense":
                return "switch_to_hybrid"

            if current_strategy == "hybrid":
                return "switch_to_bm25"

            return "switch_to_hybrid"

        if query_type == "semantic":

            if current_strategy == "bm25":
                return "switch_to_hybrid"

            if current_strategy == "hybrid":
                return "switch_to_dense"

            return "switch_to_hybrid"

        if query_type == "comparison":

            if current_strategy == "dense":
                return "switch_to_hybrid"

            if current_strategy == "bm25":
                return "switch_to_hybrid"

            if current_strategy == "hybrid":
                return "switch_to_dense"

            return "switch_to_hybrid"

        if query_type == "ambiguous":

            if current_strategy == "dense":
                return "switch_to_hybrid"

            if current_strategy == "bm25":
                return "switch_to_hybrid"

            return "switch_to_dense"

        if current_strategy == "dense":
            return "switch_to_hybrid"

        if current_strategy == "bm25":
            return "switch_to_hybrid"

        return "switch_to_dense"

    def _diagnostic_action(
        self,
        context,
        diagnosis
    ):
        plan = context.retrieval_plan

        query_analysis = context.query_analysis or {}

        query_type = query_analysis.get(
            "query_type",
            "ambiguous"
        )

        current_strategy = plan.strategy.value
        current_top_k = plan.top_k

        previous_actions = self._previous_actions(
            context
        )

        strategy_mismatch_diagnoses = {
            "lexical_strategy_mismatch",
            "semantic_strategy_mismatch",
            "ambiguous_strategy_risk",
            "comparison_strategy_mismatch",
            "retrieval_disagreement"
        }

        if diagnosis in strategy_mismatch_diagnoses:

            strategy_action = self._strategy_action(
                query_type=query_type,
                current_strategy=current_strategy
            )

            if (
                strategy_action != "keep"
                and
                strategy_action not in previous_actions
            ):
                return strategy_action

        if diagnosis in {
            "coverage_gap",
            "ranking_uncertainty",
            "comparison_coverage_risk"
        }:

            next_top_k = self._next_larger_top_k(
                current_top_k
            )

            if next_top_k is not None:

                action = f"set_top_k_{next_top_k}"

                if action not in previous_actions:
                    return action

        strategy_action = self._strategy_action(
            query_type=query_type,
            current_strategy=current_strategy
        )

        if (
            strategy_action != "keep"
            and
            strategy_action not in previous_actions
        ):
            return strategy_action

        next_top_k = self._next_larger_top_k(
            current_top_k
        )

        if next_top_k is not None:

            action = f"set_top_k_{next_top_k}"

            if action not in previous_actions:
                return action

        return "keep"

    def _policy_action(
        self,
        context,
        query_type,
        current_strategy,
        confidence_bucket,
        current_top_k
    ):
        result = self.policy.get_strategy_action(
            query_type=query_type,
            current_strategy=current_strategy,
            confidence_bucket=confidence_bucket,
            top_k=current_top_k
        )

        if result is None:
            return None

        action = result.get(
            "selected_action",
            "keep"
        )

        if action == "keep":
            return None

        previous_actions = self._previous_actions(
            context
        )

        if action in previous_actions:
            return None

        candidate = result.get(
            "candidates",
            {}
        ).get(
            action,
            {}
        )

        return {
            "action": action,
            "utility": candidate.get(
                "average_utility",
                0.0
            ),
            "support": candidate.get(
                "count",
                0
            )
        }

    def _action_cost(self, action):
        if action.startswith("set_top_k_"):
            return self.ACTION_COSTS[
                "increase_top_k"
            ]

        if action.startswith("switch_to_"):
            return self.ACTION_COSTS[
                "switch_strategy"
            ]

        return self.ACTION_COSTS[
            "keep"
        ]

    def _expected_improvement(
        self,
        diagnosis,
        action
    ):
        if action == "keep":
            return 0.0

        if diagnosis == "coverage_gap":

            if action.startswith("set_top_k_"):
                return 0.10

            return 0.04

        if diagnosis == "ranking_uncertainty":

            if action.startswith("set_top_k_"):
                return 0.07

            return 0.05

        if diagnosis == "retrieval_disagreement":

            if action == "switch_to_hybrid":
                return 0.12

            return 0.06

        if diagnosis == "lexical_strategy_mismatch":

            if action in {
                "switch_to_bm25",
                "switch_to_hybrid"
            }:
                return 0.10

            return 0.04

        if diagnosis == "semantic_strategy_mismatch":

            if action in {
                "switch_to_dense",
                "switch_to_hybrid"
            }:
                return 0.10

            return 0.04

        if diagnosis == "ambiguous_strategy_risk":

            if action == "switch_to_hybrid":
                return 0.10

            return 0.05

        if diagnosis == "comparison_strategy_mismatch":

            if action == "switch_to_hybrid":
                return 0.10

            return 0.05

        if diagnosis == "comparison_coverage_risk":

            if action.startswith("set_top_k_"):
                return 0.10

            return 0.05

        if diagnosis in {
            "no_evidence",
            "very_weak_evidence"
        }:

            if action.startswith("switch_to_"):
                return 0.12

            return 0.08

        return 0.05

    def run(self, context):
        plan = context.retrieval_plan
        evidence = context.evidence_result

        if plan is None:
            raise RuntimeError(
                "Feedback requires a retrieval plan."
            )

        if evidence is None:
            raise RuntimeError(
                "Feedback requires an evidence result."
            )

        query_analysis = context.query_analysis or {}

        query_type = query_analysis.get(
            "query_type",
            "ambiguous"
        )

        current_strategy = plan.strategy.value
        current_top_k = plan.top_k

        confidence = max(
            0.0,
            min(
                1.0,
                float(evidence.confidence)
            )
        )

        features = (
            self.feature_extractor.extract(
                context
            )
        )

        confidence_bucket = (
            self._confidence_bucket(
                confidence
            )
        )

        diagnosis, diagnosis_reason = (
            self._diagnose(
                context=context,
                features=features,
                confidence=confidence
            )
        )

        policy_candidate = (
            self._policy_action(
                context=context,
                query_type=query_type,
                current_strategy=current_strategy,
                confidence_bucket=confidence_bucket,
                current_top_k=current_top_k
            )
        )

        diagnostic_action = (
            self._diagnostic_action(
                context=context,
                diagnosis=diagnosis
            )
        )

        selected_action = diagnostic_action
        source = "diagnostic_controller"
        reason = diagnosis_reason

        if policy_candidate is not None:

            policy_action = (
                policy_candidate["action"]
            )

            policy_support = (
                policy_candidate["support"]
            )

            if policy_support >= 5:

                if diagnosis in {
                    "retrieval_disagreement",
                    "lexical_strategy_mismatch",
                    "semantic_strategy_mismatch",
                    "ambiguous_strategy_risk",
                    "comparison_strategy_mismatch"
                }:

                    selected_action = (
                        policy_action
                    )

                    source = (
                        "policy_guided_diagnosis"
                    )

                    reason = (
                        "The frozen policy provided "
                        "a supported action consistent "
                        "with the diagnosed retrieval "
                        "failure."
                    )

        expected_improvement = (
            self._expected_improvement(
                diagnosis=diagnosis,
                action=selected_action
            )
        )

        action_cost = (
            self._action_cost(
                selected_action
            )
        )

        should_retry = (
            selected_action != "keep"
            and
            expected_improvement
            >= self.minimum_improvement
        )

        target_strategy = None
        target_top_k = None

        if selected_action.startswith(
            "switch_to_"
        ):

            target_strategy = (
                selected_action[
                    len("switch_to_"):
                ]
            )

            if target_strategy == current_strategy:

                selected_action = "keep"
                should_retry = False
                expected_improvement = 0.0
                reason = (
                    "The selected strategy matches "
                    "the current strategy."
                )
                source = (
                    "diagnostic_controller"
                )
                target_strategy = None

        elif selected_action.startswith(
            "set_top_k_"
        ):

            target_top_k = int(
                selected_action[
                    len("set_top_k_"):
                ]
            )

            if target_top_k <= current_top_k:

                selected_action = "keep"
                should_retry = False
                expected_improvement = 0.0
                reason = (
                    "The selected Top-K does not "
                    "expand the current retrieval breadth."
                )
                source = (
                    "diagnostic_controller"
                )
                target_top_k = None

        decision = FeedbackDecision(
            action=selected_action,
            reason=reason,
            confidence=confidence,
            source=source,
            target_strategy=target_strategy,
            target_top_k=target_top_k,
            diagnosis=diagnosis,
            expected_improvement=expected_improvement,
            action_cost=action_cost,
            should_retry=should_retry,
            previous_confidence=getattr(
                context,
                "previous_evidence_confidence",
                None
            ),
            metadata={
                "query_type": query_type,
                "current_strategy": current_strategy,
                "current_top_k": current_top_k,
                "confidence_bucket": confidence_bucket,
                "evidence_features": features
            }
        )

        context.feedback_decision = decision

        history = self._history(
            context
        )

        history.append({
            "action": selected_action,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "expected_improvement": expected_improvement,
            "action_cost": action_cost
        })

        context.add_event(
            "feedback_decision_created"
        )

        return context