from src.core.component import Component
from src.planning.decision_engine import DecisionEngine
from src.assessment.evidence_assessor import EvidenceAssessor
from src.adaptation.feedback_controller import (
    FeedbackController
)
from src.planning.decision_types import RetrievalStrategy


class AdaptiveRetrievalOrchestrator(Component):
    """
    Executes the complete adaptive retrieval loop
    while preserving initial and final retrieval
    decisions.
    """

    def __init__(
        self,
        adaptive_retriever,
        decision_engine=None,
        evidence_assessor=None,
        feedback_controller=None,
        max_retries: int = 2
    ):
        self.adaptive_retriever = (
            adaptive_retriever
        )

        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else DecisionEngine()
        )

        self.evidence_assessor = (
            evidence_assessor
            if evidence_assessor is not None
            else EvidenceAssessor()
        )

        self.feedback_controller = (
            feedback_controller
            if feedback_controller is not None
            else FeedbackController()
        )

        self.max_retries = max_retries

    def run(self, context):

        context.add_event(
            "adaptive_retrieval_started"
        )

        retry_count = 0

        context = self.decision_engine.run(
            context
        )

        initial_plan = context.retrieval_plan

        context.decision_report[
            "initial_strategy"
        ] = initial_plan.strategy.value

        context.decision_report[
            "initial_top_k"
        ] = initial_plan.top_k

        context.decision_report[
            "initial_planner_confidence"
        ] = self._planner_confidence_value(
            initial_plan
        )

        context.decision_report[
            "attempt_history"
        ] = []

        context.decision_report[
            "strategy_transitions"
        ] = []

        context.add_event(
            "initial_retrieval_plan_created"
        )

        while retry_count <= self.max_retries:

            attempt_number = (
                retry_count + 1
            )

            plan = context.retrieval_plan

            context.add_event(
                f"retrieval_attempt_{attempt_number}"
            )

            context = self.adaptive_retriever.run(
                context
            )

            context = self.evidence_assessor.run(
                context
            )

            evidence = context.evidence_result

            attempt_record = {
                "attempt_number":
                    attempt_number,

                "strategy":
                    plan.strategy.value,

                "top_k":
                    plan.top_k,

                "evidence_confidence":
                    evidence.confidence,

                "evidence_accepted":
                    evidence.accepted
            }

            context.decision_report[
                "attempt_history"
            ].append(
                attempt_record
            )

            if evidence.accepted:

                context.add_event(
                    "retrieval_accepted"
                )

                context.decision_report[
                    "retrieval_attempts"
                ] = attempt_number

                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "accepted"

                context.decision_report[
                    "final_strategy"
                ] = plan.strategy.value

                context.decision_report[
                    "final_top_k"
                ] = plan.top_k

                context.decision_report[
                    "final_evidence_confidence"
                ] = evidence.confidence

                return context

            if retry_count >= self.max_retries:

                context.add_event(
                    "retrieval_retry_limit_reached"
                )

                context.decision_report[
                    "retrieval_attempts"
                ] = attempt_number

                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "failed_after_retries"

                context.decision_report[
                    "final_strategy"
                ] = plan.strategy.value

                context.decision_report[
                    "final_top_k"
                ] = plan.top_k

                context.decision_report[
                    "final_evidence_confidence"
                ] = evidence.confidence

                return context

            context = (
                self.feedback_controller.run(
                    context
                )
            )

            feedback = context.feedback_decision

            if not feedback.should_retry:

                context.add_event(
                    "feedback_controller_stopped_retry"
                )

                context.decision_report[
                    "retrieval_attempts"
                ] = attempt_number

                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "stopped_by_feedback"

                context.decision_report[
                    "final_strategy"
                ] = plan.strategy.value

                context.decision_report[
                    "final_top_k"
                ] = plan.top_k

                context.decision_report[
                    "final_evidence_confidence"
                ] = evidence.confidence

                return context

            old_strategy = plan.strategy.value

            self._apply_feedback(
                context
            )

            new_strategy = (
                context.retrieval_plan.strategy.value
            )

            if old_strategy != new_strategy:

                transition = {
                    "attempt_number":
                        attempt_number,

                    "old_strategy":
                        old_strategy,

                    "new_strategy":
                        new_strategy,

                    "reason":
                        feedback.reason
                }

                context.decision_report[
                    "strategy_transitions"
                ].append(
                    transition
                )

            retry_count += 1

        return context

    def _apply_feedback(self, context):

        feedback = context.feedback_decision

        plan = context.retrieval_plan

        plan.top_k = feedback.new_top_k

        plan.decision_trace.append(
            f"Feedback changed Top-K to "
            f"{plan.top_k}"
        )

        if feedback.change_strategy:

            old_strategy = plan.strategy.value

            plan.strategy = RetrievalStrategy(
                feedback.new_strategy
            )

            plan.decision_trace.append(
                "Feedback changed strategy "
                f"from {old_strategy} "
                f"to {plan.strategy.value}"
            )

        if feedback.rewrite_query:

            plan.rewrite_query = True

            plan.decision_trace.append(
                "Feedback enabled query rewriting"
            )

        if feedback.rerank:

            plan.rerank = True

            plan.decision_trace.append(
                "Feedback enabled reranking"
            )

        context.add_event(
            "retrieval_plan_adapted"
        )

    @staticmethod
    def _planner_confidence_value(
        plan
    ):

        results = (
            plan.policy_results
        )

        if not results:
            return 0.0

        values = [
            result.confidence
            for result in results.values()
        ]

        return sum(values) / len(values)