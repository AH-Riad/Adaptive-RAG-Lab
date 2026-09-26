import ast
import json
from pathlib import Path

from src.core.component import Component
from src.adaptation.feedback_decision import FeedbackDecision
from src.assessment.evidence_features import EvidenceFeatureExtractor


class FeedbackController(Component):
    SUPPORTED_TOP_K = (3, 5, 10, 15)

    ACTION_COSTS = {
        "keep": 0.0,
        "increase_top_k": 0.20,
        "switch_strategy": 0.40
    }

    STRATEGY_DIAGNOSES = {
        "retrieval_disagreement",
        "lexical_strategy_mismatch",
        "semantic_strategy_mismatch",
        "ambiguous_strategy_risk"
    }

    TOP_K_DIAGNOSES = {
        "coverage_gap",
        "ranking_uncertainty",
        "comparison_coverage_risk"
    }

    def __init__(
        self,
        policy_path: str,
        minimum_samples: int = 5,
        minimum_improvement: float = 0.03
    ):
        self.policy_path = Path(policy_path)
        self.minimum_samples = minimum_samples
        self.minimum_improvement = minimum_improvement
        self.feature_extractor = EvidenceFeatureExtractor()
        self.policy = self._load_policy()

    def _load_policy(self):
        if not self.policy_path.exists():
            return None

        with self.policy_path.open("r", encoding="utf-8") as file:
            return json.load(file)

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
        history = getattr(context, "feedback_history", None)
        if history is None:
            history = []
            context.feedback_history = history
        return history

    def _previous_actions(self, context):
        return {
            item.get("action")
            for item in self._history(context)
            if item.get("action")
        }

    def _policy_state(self, query_type, current_strategy, confidence_bucket, top_k):
        return str(
            (
                query_type,
                current_strategy,
                confidence_bucket,
                top_k
            )
        )

    def _get_policy_record(
        self,
        section,
        query_type,
        current_strategy,
        confidence_bucket,
        current_top_k
    ):
        if not self.policy:
            return None

        policy_section = self.policy.get(section, {})
        state = self._policy_state(
            query_type=query_type,
            current_strategy=current_strategy,
            confidence_bucket=confidence_bucket,
            top_k=current_top_k
        )
        return policy_section.get(state)

    def _policy_candidate(self, record, current_top_k):
        if record is None:
            return None

        selected_action = record.get("selected_action", "keep")
        candidates = record.get("candidates", {})
        candidate = candidates.get(selected_action, {})
        support = candidate.get("count", 0)

        if support < self.minimum_samples:
            return None

        if selected_action.startswith("set_top_k_"):
            try:
                target_top_k = int(selected_action[len("set_top_k_"):])
            except ValueError:
                return None

            if target_top_k <= current_top_k:
                return {
                    "action": "keep",
                    "support": support,
                    "utility": candidate.get("average_utility", 0.0),
                    "source": "frozen_topk_policy",
                    "policy_action": selected_action
                }

        return {
            "action": selected_action,
            "support": support,
            "utility": candidate.get("average_utility", 0.0),
            "source": "frozen_policy",
            "policy_action": selected_action
        }

    def _learned_action(
        self,
        context,
        query_type,
        current_strategy,
        confidence_bucket,
        current_top_k,
        diagnosis
    ):
        strategy_record = self._get_policy_record(
            section="strategy_policy",
            query_type=query_type,
            current_strategy=current_strategy,
            confidence_bucket=confidence_bucket,
            current_top_k=current_top_k
        )

        topk_record = self._get_policy_record(
            section="topk_policy",
            query_type=query_type,
            current_strategy=current_strategy,
            confidence_bucket=confidence_bucket,
            current_top_k=current_top_k
        )

        if diagnosis in self.STRATEGY_DIAGNOSES:
            candidate = self._policy_candidate(
                strategy_record,
                current_top_k
            )
            if candidate is not None:
                return candidate

        if diagnosis in self.TOP_K_DIAGNOSES:
            candidate = self._policy_candidate(
                topk_record,
                current_top_k
            )
            if candidate is not None:
                return candidate

        strategy_candidate = self._policy_candidate(
            strategy_record,
            current_top_k
        )
        topk_candidate = self._policy_candidate(
            topk_record,
            current_top_k
        )

        candidates = [
            candidate
            for candidate in (strategy_candidate, topk_candidate)
            if candidate is not None
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: item["utility"]
        )

    def _diagnose(self, context, features, confidence):
        plan = context.retrieval_plan
        evidence = context.evidence_result

        query_analysis = context.query_analysis or {}
        query_type = query_analysis.get("query_type", "ambiguous")
        strategy = plan.strategy.value
        top_k = plan.top_k

        if evidence.retrieved_count == 0:
            return "no_evidence", "No evidence was retrieved."

        disagreement = features.get("dense_bm25_agreement")
        top1 = features.get("top1_score", 0.0)
        top1_top2_gap = features.get("top1_top2_gap", 0.0)
        score_coverage = getattr(evidence, "coverage", 0.0)

        if disagreement is not None and 0.0 < disagreement < 0.50:
            return (
                "retrieval_disagreement",
                "Dense and lexical retrieval signals disagree strongly."
            )

        if score_coverage < 0.50 and evidence.retrieved_count >= 3:
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

    def _strategy_action(self, query_type, current_strategy):
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
            if current_strategy == "hybrid":
                return "switch_to_dense"
            if current_strategy == "dense":
                return "switch_to_hybrid"
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

    def _diagnostic_action(self, context, diagnosis):
        plan = context.retrieval_plan
        query_analysis = context.query_analysis or {}
        query_type = query_analysis.get("query_type", "ambiguous")
        current_strategy = plan.strategy.value
        current_top_k = plan.top_k
        previous_actions = self._previous_actions(context)

        if diagnosis in self.TOP_K_DIAGNOSES:
            next_top_k = self._next_larger_top_k(current_top_k)
            if next_top_k is not None:
                action = f"set_top_k_{next_top_k}"
                if action not in previous_actions:
                    return action

        strategy_action = self._strategy_action(
            query_type=query_type,
            current_strategy=current_strategy
        )

        if strategy_action != "keep" and strategy_action not in previous_actions:
            return strategy_action

        next_top_k = self._next_larger_top_k(current_top_k)
        if next_top_k is not None:
            action = f"set_top_k_{next_top_k}"
            if action not in previous_actions:
                return action

        return "keep"

    def _expected_improvement(self, diagnosis, action):
        if action == "keep":
            return 0.0

        if diagnosis == "coverage_gap":
            return 0.10 if action.startswith("set_top_k_") else 0.04

        if diagnosis == "ranking_uncertainty":
            return 0.07 if action.startswith("set_top_k_") else 0.05

        if diagnosis == "retrieval_disagreement":
            return 0.12 if action == "switch_to_hybrid" else 0.06

        if diagnosis == "lexical_strategy_mismatch":
            return 0.10 if action in {"switch_to_bm25", "switch_to_hybrid"} else 0.04

        if diagnosis == "semantic_strategy_mismatch":
            return 0.10 if action in {"switch_to_dense", "switch_to_hybrid"} else 0.04

        if diagnosis == "ambiguous_strategy_risk":
            return 0.10 if action == "switch_to_hybrid" else 0.05

        if diagnosis == "comparison_coverage_risk":
            return 0.10 if action.startswith("set_top_k_") else 0.05

        if diagnosis in {"no_evidence", "very_weak_evidence"}:
            return 0.12 if action.startswith("switch_to_") else 0.08

        return 0.05

    def _validate_action(self, action, current_strategy, current_top_k):
        if action == "keep":
            return True

        if action.startswith("switch_to_"):
            target_strategy = action[len("switch_to_"):]
            return target_strategy in {"dense", "bm25", "hybrid"} and target_strategy != current_strategy

        if action.startswith("set_top_k_"):
            try:
                target_top_k = int(action[len("set_top_k_"):])
            except ValueError:
                return False
            return target_top_k > current_top_k and target_top_k in self.SUPPORTED_TOP_K

        return False

    def run(self, context):
        plan = context.retrieval_plan
        evidence = context.evidence_result

        if plan is None:
            raise RuntimeError("Feedback requires a retrieval plan.")
        if evidence is None:
            raise RuntimeError("Feedback requires an evidence result.")

        query_analysis = context.query_analysis or {}
        query_type = query_analysis.get("query_type", "ambiguous")
        current_strategy = plan.strategy.value
        current_top_k = plan.top_k

        confidence = max(0.0, min(1.0, float(evidence.confidence)))
        features = self.feature_extractor.extract(context)
        confidence_bucket = self._confidence_bucket(confidence)

        diagnosis, diagnosis_reason = self._diagnose(
            context=context,
            features=features,
            confidence=confidence
        )

        learned_candidate = self._learned_action(
            context=context,
            query_type=query_type,
            current_strategy=current_strategy,
            confidence_bucket=confidence_bucket,
            current_top_k=current_top_k,
            diagnosis=diagnosis
        )

        diagnostic_action = self._diagnostic_action(
            context=context,
            diagnosis=diagnosis
        )

        if learned_candidate is not None:
            selected_action = learned_candidate["action"]
            source = learned_candidate["source"]
            reason = (
                "The frozen dev policy has sufficient support for this state; "
                "its selected action is authoritative."
            )
        else:
            selected_action = diagnostic_action
            source = "diagnostic_fallback"
            reason = diagnosis_reason

        if not self._validate_action(
            action=selected_action,
            current_strategy=current_strategy,
            current_top_k=current_top_k
        ):
            selected_action = "keep"
            source = "safety_fallback"
            reason = "The selected policy or diagnostic action was invalid for the current retrieval state."

        previous_actions = self._previous_actions(context)
        if selected_action in previous_actions:
            selected_action = "keep"
            source = "retry_guard"
            reason = "The selected action was already attempted for this query."

        expected_improvement = self._expected_improvement(
            diagnosis=diagnosis,
            action=selected_action
        )
        action_cost = self._action_cost(selected_action)
        should_retry = (
            selected_action != "keep"
            and expected_improvement >= self.minimum_improvement
        )

        target_strategy = None
        target_top_k = None

        if selected_action.startswith("switch_to_"):
            target_strategy = selected_action[len("switch_to_"):]

        elif selected_action.startswith("set_top_k_"):
            target_top_k = int(selected_action[len("set_top_k_"):])

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
                "evidence_features": features,
                "policy_available": self.policy is not None
            }
        )

        context.feedback_decision = decision
        history = self._history(context)
        history.append({
            "action": selected_action,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "expected_improvement": expected_improvement,
            "action_cost": action_cost,
            "source": source
        })

        context.add_event("feedback_decision_created")
        return context

    def _action_cost(self, action):
        if action.startswith("set_top_k_"):
            return self.ACTION_COSTS["increase_top_k"]
        if action.startswith("switch_to_"):
            return self.ACTION_COSTS["switch_strategy"]
        return self.ACTION_COSTS["keep"]
