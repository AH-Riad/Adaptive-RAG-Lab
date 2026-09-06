from src.core.component import Component
from src.planning.decision_engine import DecisionEngine
from src.assessment.evidence_assessor import EvidenceAssessor
from src.adaptation.feedback_controller import FeedbackController

class AdaptiveRetrievalOrchestrator(Component):
    """
    Executes the complete adaptive retrieval loop
    with closed-loop feedback and trajectory tracking.
    """
    def __init__(
        self,
        adaptive_retriever,
        decision_engine=None,
        evidence_assessor=None,
        feedback_controller=None,
        max_retries: int = 2
    ):
        self.adaptive_retriever = adaptive_retriever

        self.decision_engine = (
            decision_engine if decision_engine is not None else DecisionEngine()
        )

        self.evidence_assessor = (
            evidence_assessor if evidence_assessor is not None else EvidenceAssessor(
                calibrated_model_path="results/logs/fiqa_dev_evidence_calibrator_v1.json"
            )
        )

        self.feedback_controller = (
            feedback_controller if feedback_controller is not None else FeedbackController(
                policy_path="results/logs/fiqa_dev_action_policy_v5_1.json",
                minimum_samples=5
            )
        )

        self.max_retries = max_retries

    def run(self, context):
        context.add_event("adaptive_retrieval_started")
        retry_count = 0

        context = self.decision_engine.run(context)
        initial_plan = context.retrieval_plan

        context.decision_report["initial_strategy"] = initial_plan.strategy.value
        context.decision_report["initial_top_k"] = initial_plan.top_k
        context.decision_report["initial_planner_confidence"] = self._planner_confidence_value(initial_plan)
        context.decision_report["attempt_history"] = []
        context.decision_report["strategy_transitions"] = []
        context.decision_report["feedback_history"] = []
        
        context.feedback_history = []
        context.add_event("initial_retrieval_plan_created")

        while retry_count <= self.max_retries:
            attempt_number = retry_count + 1
            plan = context.retrieval_plan

            context.add_event(f"retrieval_attempt_{attempt_number}")

            context = self.adaptive_retriever.run(context)
            context = self.evidence_assessor.run(context)
            evidence = context.evidence_result

            previous_confidence = getattr(context, "previous_evidence_confidence", None)
            confidence_change = None

            if previous_confidence is not None:
                confidence_change = evidence.confidence - previous_confidence

            attempt_record = {
                "attempt_number": attempt_number,
                "strategy": plan.strategy.value,
                "top_k": plan.top_k,
                "evidence_confidence": evidence.confidence,
                "evidence_accepted": evidence.accepted,
                "confidence_change": confidence_change
            }

            context.decision_report["attempt_history"].append(attempt_record)

            if evidence.accepted:
                context.add_event("retrieval_accepted")
                context.decision_report["retrieval_attempts"] = attempt_number
                context.decision_report["adaptive_retrieval_status"] = "accepted"
                context.decision_report["final_strategy"] = plan.strategy.value
                context.decision_report["final_top_k"] = plan.top_k
                context.decision_report["final_evidence_confidence"] = evidence.confidence
                return context

            if retry_count >= self.max_retries:
                context.add_event("retrieval_retry_limit_reached")
                context.decision_report["retrieval_attempts"] = attempt_number
                context.decision_report["adaptive_retrieval_status"] = "failed_after_retries"
                context.decision_report["final_strategy"] = plan.strategy.value
                context.decision_report["final_top_k"] = plan.top_k
                context.decision_report["final_evidence_confidence"] = evidence.confidence
                return context

            context.previous_evidence_confidence = evidence.confidence

            context = self.feedback_controller.run(context)
            feedback = context.feedback_decision

            context.decision_report["feedback_history"].append(feedback.to_dict())

            if not feedback.should_retry:
                context.add_event("feedback_controller_stopped_retry")
                context.decision_report["retrieval_attempts"] = attempt_number
                context.decision_report["adaptive_retrieval_status"] = "stopped_by_feedback"
                context.decision_report["final_strategy"] = plan.strategy.value
                context.decision_report["final_top_k"] = plan.top_k
                context.decision_report["final_evidence_confidence"] = evidence.confidence
                return context

            old_strategy = plan.strategy.value
            old_top_k = plan.top_k

            self._apply_feedback(context)

            new_strategy = context.retrieval_plan.strategy.value
            new_top_k = context.retrieval_plan.top_k

            if old_strategy != new_strategy:
                transition = {
                    "attempt_number": attempt_number,
                    "old_strategy": old_strategy,
                    "new_strategy": new_strategy,
                    "old_top_k": old_top_k,
                    "new_top_k": new_top_k,
                    "diagnosis": feedback.diagnosis,
                    "reason": feedback.reason
                }
                context.decision_report["strategy_transitions"].append(transition)
            elif old_top_k != new_top_k:
                context.add_event("retrieval_top_k_changed")

            retry_count += 1

        return context

    def _apply_feedback(self, context):
        feedback = context.feedback_decision
        plan = context.retrieval_plan

        old_strategy = plan.strategy.value
        old_top_k = plan.top_k
        action = feedback.action

        if action == "keep":
            plan.decision_trace.append("Feedback controller selected KEEP.")
            context.add_event("retrieval_plan_unchanged")
            return

        if action.startswith("switch_to_"):
            target_strategy = feedback.target_strategy

            if target_strategy is None:
                context.add_event("invalid_feedback_action")
                return

            if target_strategy == old_strategy:
                plan.decision_trace.append(
                    "Feedback requested the current strategy; no change applied."
                )
                context.add_event("retrieval_plan_unchanged")
                return

            plan.strategy = type(plan.strategy)(target_strategy)
            plan.decision_trace.append(
                f"Feedback changed strategy from {old_strategy} to {target_strategy}."
            )
            context.add_event("retrieval_strategy_changed")
            return

        if action.startswith("set_top_k_"):
            target_top_k = feedback.target_top_k

            if target_top_k is None:
                context.add_event("invalid_feedback_action")
                return

            if target_top_k <= old_top_k:
                plan.decision_trace.append(
                    "Feedback requested a non-expanding Top-K; no change applied."
                )
                context.add_event("retrieval_plan_unchanged")
                return

            plan.top_k = target_top_k
            plan.decision_trace.append(
                f"Feedback changed Top-K from {old_top_k} to {target_top_k}."
            )
            context.add_event("retrieval_top_k_changed")
            return

        plan.decision_trace.append(f"Unsupported feedback action: {action}")
        context.add_event("invalid_feedback_action")

    @staticmethod
    def _planner_confidence_value(plan):
        results = plan.policy_results
        if not results:
            return 0.0

        values = [result.confidence for result in results.values()]
        return sum(values) / len(values)