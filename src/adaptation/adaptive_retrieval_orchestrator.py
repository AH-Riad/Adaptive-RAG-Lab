from src.core.component import Component
from src.planning.decision_engine import DecisionEngine
from src.assessment.evidence_assessor import EvidenceAssessor
from src.adaptation.feedback_controller import FeedbackController
from src.planning.decision_types import RetrievalStrategy

class AdaptiveRetrievalOrchestrator(Component):
    """
    Coordinates the complete adaptive retrieval loop.

    Flow:

        Query
          ↓
        Decision Engine
          ↓
        Retrieval
          ↓
        Evidence Assessment
          ↓
        Feedback Controller
          ↓
        Retry / Accept
    """

    def __init__(
        self,
        retriever,
        decision_engine=None,
        evidence_assessor=None,
        feedback_controller=None,
        max_retries: int = 2,
    ):
        self.retriever = retriever

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

        # INITIAL PLANNING

        context = self.decision_engine.run(context)

        context.add_event(
            "initial_retrieval_plan_created"
        )

        # ADAPTIVE RETRIEVAL LOOP

        while retry_count <= self.max_retries:

            plan = context.retrieval_plan

            # Apply the current retrieval configuration

            self._configure_retriever(plan)

            context.add_event(
                f"retrieval_attempt_{retry_count + 1}"
            )

            # Retrieve

            retrieval_result = self.retriever.retrieve(
                context.query
            )

            context.retrieval_result = retrieval_result

            # Assess evidence

            context = self.evidence_assessor.run(
                context
            )

            evidence = context.evidence_result

            # Successful retrieval

            if evidence.accepted:

                context.add_event(
                    "retrieval_accepted"
                )

                context.decision_report[
                    "retrieval_attempts"
                ] = retry_count + 1

                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "accepted"

                return context

            # Maximum retries reached

            if retry_count >= self.max_retries:

                context.add_event(
                    "retrieval_retry_limit_reached"
                )

                context.decision_report[
                    "retrieval_attempts"
                ] = retry_count + 1

                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "failed_after_retries"

                return context

            # Generate adaptation decision

            context = self.feedback_controller.run(
                context
            )

            feedback = context.feedback_decision

            if not feedback.should_retry:

                context.add_event(
                    "feedback_controller_stopped_retry"
                )

                context.decision_report[
                    "retrieval_attempts"
                ] = retry_count + 1

                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "stopped_by_feedback"

                return context

            # Apply feedback to current retrieval plan

            self._apply_feedback(
                context
            )

            retry_count += 1

        return context

    # RETRIEVER CONFIGURATION

    def _configure_retriever(self, plan):

        if hasattr(self.retriever, "top_k"):

            self.retriever.top_k = plan.top_k

    # APPLY ADAPTIVE FEEDBACK

    def _apply_feedback(self, context):

        feedback = context.feedback_decision
        plan = context.retrieval_plan

        # Top-K adaptation

        plan.top_k = feedback.new_top_k

        plan.decision_trace.append(
            f"Feedback increased Top-K to {plan.top_k}"
        )

        # Retrieval strategy adaptation

        if feedback.change_strategy:

            old_strategy = plan.strategy

            plan.strategy = RetrievalStrategy(
                feedback.new_strategy
            )
            
            plan.decision_trace.append(
                "Feedback changed retrieval strategy "
                f"from {old_strategy} to {feedback.new_strategy}"
            )

        # Query rewriting

        if feedback.rewrite_query:

            plan.rewrite_query = True

            plan.decision_trace.append(
                "Feedback enabled query rewriting"
            )

        # Reranking

        if feedback.rerank:

            plan.rerank = True

            plan.decision_trace.append(
                "Feedback enabled reranking"
            )

        context.add_event(
            "retrieval_plan_adapted"
        )