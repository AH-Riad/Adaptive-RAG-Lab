import json
from pathlib import Path

from src.core.component import Component
from src.adaptation.feedback_decision import FeedbackDecision
from src.assessment.evidence_features import EvidenceFeatureExtractor


class FeedbackController(Component):
    """Runtime controller for the frozen diagnosis-first D²RAG policy."""

    SUPPORTED_TOP_K = (3, 5, 8, 10, 15)

    ACTION_COSTS = {
        "keep": 0.0,
        "increase_top_k": 0.20,
        "switch_strategy": 0.40,
    }

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

    EXACT_MIN_SUPPORT = 8
    QUERY_TYPE_BACKOFF_MIN_SUPPORT = 20
    STRATEGY_BACKOFF_MIN_SUPPORT = 30
    DIAGNOSIS_BACKOFF_MIN_SUPPORT = 40

    def __init__(
        self,
        policy_path: str,
        minimum_samples: int = 8,
        minimum_improvement: float = 0.02,
    ):
        self.policy_path = Path(policy_path)
        self.minimum_samples = int(minimum_samples)
        self.minimum_improvement = float(minimum_improvement)
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

    @staticmethod
    def _exact_state(query_type, strategy, confidence_bucket, diagnosis, top_k):
        return str((query_type, strategy, confidence_bucket, diagnosis, top_k))

    @staticmethod
    def _query_type_backoff_state(query_type, strategy, diagnosis, top_k):
        return str((query_type, strategy, diagnosis, top_k))

    @staticmethod
    def _strategy_backoff_state(strategy, diagnosis, top_k):
        return str((strategy, diagnosis, top_k))

    @staticmethod
    def _diagnosis_backoff_state(diagnosis, top_k):
        return str((diagnosis, top_k))

    def _record_is_supported(self, record, minimum_support):
        if record is None:
            return False
        samples = int(record.get("samples", 0))
        if samples < minimum_support:
            return False
        return "selected_action" in record

    def _get_hierarchical_record(
        self,
        section,
        query_type,
        current_strategy,
        confidence_bucket,
        diagnosis,
        current_top_k,
    ):
        if not self.policy:
            return None

        exact = self.policy.get(section, {}).get(
            self._exact_state(
                query_type,
                current_strategy,
                confidence_bucket,
                diagnosis,
                current_top_k,
            )
        )
        if self._record_is_supported(exact, self.EXACT_MIN_SUPPORT):
            return exact, "exact"

        query_type_section = self.policy.get(
            f"{section.replace('_policy', '')}_backoff_query_type",
            {},
        )
        query_type_record = query_type_section.get(
            self._query_type_backoff_state(
                query_type,
                current_strategy,
                diagnosis,
                current_top_k,
            )
        )
        if self._record_is_supported(
            query_type_record,
            self.QUERY_TYPE_BACKOFF_MIN_SUPPORT,
        ):
            return query_type_record, "query_type"

        strategy_section = self.policy.get(
            f"{section.replace('_policy', '')}_backoff_strategy",
            {},
        )
        strategy_record = strategy_section.get(
            self._strategy_backoff_state(
                current_strategy,
                diagnosis,
                current_top_k,
            )
        )
        if self._record_is_supported(
            strategy_record,
            self.STRATEGY_BACKOFF_MIN_SUPPORT,
        ):
            return strategy_record, "strategy"

        diagnosis_section = self.policy.get(
            f"{section.replace('_policy', '')}_backoff_diagnosis",
            {},
        )
        diagnosis_record = diagnosis_section.get(
            self._diagnosis_backoff_state(
                diagnosis,
                current_top_k,
            )
        )
        if self._record_is_supported(
            diagnosis_record,
            self.DIAGNOSIS_BACKOFF_MIN_SUPPORT,
        ):
            return diagnosis_record, "diagnosis"

        return None

    def _validate_action(self, action, current_strategy, current_top_k):
        if action == "keep":
            return True

        if action.startswith("switch_to_"):
            target = action[len("switch_to_"):]
            return (
                target in {"dense", "bm25", "hybrid"}
                and target != current_strategy
            )

        if action.startswith("set_top_k_"):
            try:
                target_top_k = int(action[len("set_top_k_"):])
            except ValueError:
                return False
            return (
                target_top_k in self.SUPPORTED_TOP_K
                and target_top_k > current_top_k
            )

        return False

    def _policy_candidate(self, record_with_level, current_strategy, current_top_k):
        if record_with_level is None:
            return None

        record, level = record_with_level
        action = record.get("selected_action", "keep")
        candidate = record.get("candidates", {}).get(action, {})
        support = int(candidate.get("count", 0))

        if support < self.minimum_samples:
            return None

        if not self._validate_action(action, current_strategy, current_top_k):
            return None

        gain = float(candidate.get("average_gain", 0.0))
        win_rate = float(candidate.get("win_rate", 0.0))
        harm_rate = float(candidate.get("harm_rate", 0.0))

        if action != "keep" and gain < self.minimum_improvement:
            return None

        return {
            "action": action,
            "support": support,
            "average_gain": gain,
            "win_rate": win_rate,
            "harm_rate": harm_rate,
            "utility": float(candidate.get("average_utility", 0.0)),
            "source": f"failure_conditioned_policy_v7_{level}",
            "policy_action": action,
            "policy_level": level,
        }

    def _learned_action(
        self,
        query_type,
        current_strategy,
        confidence_bucket,
        current_top_k,
        diagnosis,
    ):
        if diagnosis in self.STRATEGY_DIAGNOSES:
            sections = ["strategy_policy"]
        elif diagnosis in self.TOP_K_DIAGNOSES:
            sections = ["topk_policy"]
        else:
            sections = ["combined_policy"]

        candidates = []
        for section in sections:
            record = self._get_hierarchical_record(
                section=section,
                query_type=query_type,
                current_strategy=current_strategy,
                confidence_bucket=confidence_bucket,
                diagnosis=diagnosis,
                current_top_k=current_top_k,
            )
            candidate = self._policy_candidate(
                record_with_level=record,
                current_strategy=current_strategy,
                current_top_k=current_top_k,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: (
                item["average_gain"],
                item["win_rate"],
                -item["harm_rate"],
                item["support"],
            ),
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

        if query_type == "lexical" and strategy == "dense":
            return "lexical_strategy_mismatch", "Lexical query routed to dense retrieval."

        if query_type == "semantic" and strategy == "bm25":
            return "semantic_strategy_mismatch", "Semantic query routed to lexical retrieval."

        if query_type == "ambiguous" and strategy == "dense":
            return "ambiguous_strategy_risk", "Ambiguous query has no lexical retrieval support."

        if query_type == "comparison" and strategy == "hybrid" and top_k < 10:
            return "comparison_coverage_risk", "Comparison query may require broader retrieval coverage."

        disagreement = features.get("dense_bm25_agreement")
        if disagreement is not None and 0.0 < disagreement < 0.50:
            return "retrieval_disagreement", "Dense and lexical retrieval signals disagree strongly."

        score_coverage = getattr(evidence, "coverage", 0.0)
        if score_coverage < 0.50 and evidence.retrieved_count >= 3:
            return "coverage_gap", "Retrieved evidence has insufficient score-based coverage."

        top1 = features.get("top1_score", 0.0)
        top1_top2_gap = features.get("top1_top2_gap", 0.0)
        if top1 >= 0.45 and top1_top2_gap < 0.05:
            return "ranking_uncertainty", "Top-ranked evidence is weakly separated from the next result."

        if confidence < 0.25:
            return "very_weak_evidence", "Calibrated evidence confidence is very low."

        if confidence < 0.50:
            return "weak_evidence", "Evidence confidence is below the medium-confidence range."

        return "uncertain_failure", "Evidence was rejected without a dominant failure pattern."

    def _action_cost(self, action):
        if action.startswith("set_top_k_"):
            return self.ACTION_COSTS["increase_top_k"]
        if action.startswith("switch_to_"):
            return self.ACTION_COSTS["switch_strategy"]
        return self.ACTION_COSTS["keep"]

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
            confidence=confidence,
        )

        learned_candidate = self._learned_action(
            query_type=query_type,
            current_strategy=current_strategy,
            confidence_bucket=confidence_bucket,
            current_top_k=current_top_k,
            diagnosis=diagnosis,
        )

        if learned_candidate is None:
            selected_action = "keep"
            source = "policy_no_supported_action"
            reason = diagnosis_reason
            policy_gain = 0.0
            policy_level = None
            policy_support = 0
            policy_win_rate = 0.0
            policy_harm_rate = 0.0
        else:
            selected_action = learned_candidate["action"]
            source = learned_candidate["source"]
            reason = "A supported frozen dev policy selected this counterfactual action."
            policy_gain = learned_candidate["average_gain"]
            policy_level = learned_candidate["policy_level"]
            policy_support = learned_candidate["support"]
            policy_win_rate = learned_candidate["win_rate"]
            policy_harm_rate = learned_candidate["harm_rate"]

        if not self._validate_action(
            selected_action,
            current_strategy,
            current_top_k,
        ):
            selected_action = "keep"
            source = "safety_fallback"
            reason = "The frozen policy action was invalid for the current retrieval state."
            policy_gain = 0.0
            policy_level = None
            policy_support = 0
            policy_win_rate = 0.0
            policy_harm_rate = 0.0

        if selected_action in self._previous_actions(context):
            selected_action = "keep"
            source = "retry_guard"
            reason = "The selected action was already attempted for this query."
            policy_gain = 0.0
            policy_level = None
            policy_support = 0
            policy_win_rate = 0.0
            policy_harm_rate = 0.0

        should_retry = (
            selected_action != "keep"
            and policy_gain >= self.minimum_improvement
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
            expected_improvement=max(0.0, policy_gain),
            action_cost=self._action_cost(selected_action),
            should_retry=should_retry,
            previous_confidence=getattr(
                context,
                "previous_evidence_confidence",
                None,
            ),
            metadata={
                "query_type": query_type,
                "current_strategy": current_strategy,
                "current_top_k": current_top_k,
                "confidence_bucket": confidence_bucket,
                "diagnosis": diagnosis,
                "evidence_features": features,
                "policy_available": self.policy is not None,
                "policy_level": policy_level,
                "policy_support": policy_support,
                "policy_gain": policy_gain,
                "policy_win_rate": policy_win_rate,
                "policy_harm_rate": policy_harm_rate,
            },
        )

        context.feedback_decision = decision
        self._history(context).append(
            {
                "action": selected_action,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "expected_improvement": max(0.0, policy_gain),
                "action_cost": self._action_cost(selected_action),
                "source": source,
                "policy_level": policy_level,
                "policy_support": policy_support,
                "policy_win_rate": policy_win_rate,
                "policy_harm_rate": policy_harm_rate,
            }
        )
        context.add_event("feedback_decision_created")
        return context
