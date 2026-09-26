from src.core.component import Component
from src.planning.decision_engine import DecisionEngine
from src.assessment.evidence_assessor import EvidenceAssessor
from src.adaptation.feedback_controller import FeedbackController


class AdaptiveRetrievalOrchestrator(Component):
    """
    Executes the adaptive retrieval loop with closed-loop feedback.

    The orchestrator also enforces Top-K propagation across the retriever
    objects before every retrieval attempt. This prevents the retrieval plan
    from requesting K=10 or K=15 while the underlying retrievers remain at a
    smaller configured value.
    """

    def __init__(
        self,
        adaptive_retriever,
        decision_engine=None,
        evidence_assessor=None,
        feedback_controller=None,
        max_retries: int = 2,
        strict_top_k_integrity: bool = True
    ):
        self.adaptive_retriever = adaptive_retriever
        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else DecisionEngine()
        )
        self.evidence_assessor = (
            evidence_assessor
            if evidence_assessor is not None
            else EvidenceAssessor(
                calibrated_model_path=(
                    "results/logs/"
                    "fiqa_dev_evidence_calibrator_v1.json"
                )
            )
        )
        self.feedback_controller = (
            feedback_controller
            if feedback_controller is not None
            else FeedbackController(
                policy_path=(
                    "results/logs/"
                    "fiqa_dev_action_policy_v5_1.json"
                ),
                minimum_samples=5
            )
        )
        self.max_retries = max_retries
        self.strict_top_k_integrity = strict_top_k_integrity

    def run(self, context):
        context.add_event("adaptive_retrieval_started")
        retry_count = 0
        context = self.decision_engine.run(context)
        initial_plan = context.retrieval_plan

        context.decision_report["initial_strategy"] = (
            initial_plan.strategy.value
        )
        context.decision_report["initial_top_k"] = (
            initial_plan.top_k
        )
        context.decision_report["initial_planner_confidence"] = (
            self._planner_confidence_value(initial_plan)
        )
        context.decision_report["attempt_history"] = []
        context.decision_report["strategy_transitions"] = []
        context.decision_report["feedback_history"] = []
        context.decision_report["retrieval_integrity"] = []
        context.feedback_history = []

        context.add_event("initial_retrieval_plan_created")

        while retry_count <= self.max_retries:
            attempt_number = retry_count + 1
            plan = context.retrieval_plan

            context.add_event(
                f"retrieval_attempt_{attempt_number}"
            )

            requested_top_k = plan.top_k
            self._sync_retriever_top_k(requested_top_k)

            context = self.adaptive_retriever.run(context)

            actual_retrieved_count = self._retrieved_count(context)
            integrity_ok = actual_retrieved_count >= requested_top_k

            integrity_record = {
                "attempt_number": attempt_number,
                "strategy": plan.strategy.value,
                "requested_top_k": requested_top_k,
                "actual_retrieved_count": actual_retrieved_count,
                "integrity_ok": integrity_ok
            }

            context.decision_report["retrieval_integrity"].append(
                integrity_record
            )

            if not integrity_ok:
                context.add_event("retrieval_top_k_integrity_failure")

                if self.strict_top_k_integrity:
                    raise RuntimeError(
                        "Top-K integrity failure: "
                        f"requested K={requested_top_k}, "
                        f"retrieved {actual_retrieved_count}. "
                        "The retrieval backend is not honoring the plan."
                    )

            context = self.evidence_assessor.run(context)
            evidence = context.evidence_result

            previous_confidence = getattr(
                context,
                "previous_evidence_confidence",
                None
            )
            confidence_change = None

            if previous_confidence is not None:
                confidence_change = (
                    evidence.confidence - previous_confidence
                )

            attempt_record = {
                "attempt_number": attempt_number,
                "strategy": plan.strategy.value,
                "top_k": plan.top_k,
                "requested_top_k": requested_top_k,
                "actual_retrieved_count": actual_retrieved_count,
                "retrieved_chunk_ids": [
                    chunk.chunk_id
                    for chunk in getattr(
                        getattr(context, "retrieval_result", None),
                        "retrieved_chunks",
                        []
                    )
                ],
                "evidence_confidence": evidence.confidence,
                "evidence_accepted": evidence.accepted,
                "confidence_change": confidence_change
            }

            context.decision_report["attempt_history"].append(
                attempt_record
            )

            if evidence.accepted:
                context.add_event("retrieval_accepted")
                context.decision_report["retrieval_attempts"] = (
                    attempt_number
                )
                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "accepted"
                context.decision_report["final_strategy"] = (
                    plan.strategy.value
                )
                context.decision_report["final_top_k"] = plan.top_k
                context.decision_report[
                    "final_evidence_confidence"
                ] = evidence.confidence
                context.decision_report[
                    "final_retrieved_count"
                ] = actual_retrieved_count
                return context

            if retry_count >= self.max_retries:
                context.add_event("retrieval_retry_limit_reached")
                context.decision_report["retrieval_attempts"] = (
                    attempt_number
                )
                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "failed_after_retries"
                context.decision_report["final_strategy"] = (
                    plan.strategy.value
                )
                context.decision_report["final_top_k"] = plan.top_k
                context.decision_report[
                    "final_evidence_confidence"
                ] = evidence.confidence
                context.decision_report[
                    "final_retrieved_count"
                ] = actual_retrieved_count
                return context

            context.previous_evidence_confidence = evidence.confidence
            context = self.feedback_controller.run(context)
            feedback = context.feedback_decision

            context.decision_report["feedback_history"].append(
                feedback.to_dict()
            )

            if not feedback.should_retry:
                context.add_event(
                    "feedback_controller_stopped_retry"
                )
                context.decision_report["retrieval_attempts"] = (
                    attempt_number
                )
                context.decision_report[
                    "adaptive_retrieval_status"
                ] = "stopped_by_feedback"
                context.decision_report["final_strategy"] = (
                    plan.strategy.value
                )
                context.decision_report["final_top_k"] = plan.top_k
                context.decision_report[
                    "final_evidence_confidence"
                ] = evidence.confidence
                context.decision_report[
                    "final_retrieved_count"
                ] = actual_retrieved_count
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
                context.decision_report["strategy_transitions"].append(
                    transition
                )
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
            plan.decision_trace.append(
                "Feedback controller selected KEEP."
            )
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
                f"Feedback changed strategy from {old_strategy} "
                f"to {target_strategy}."
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
                f"Feedback changed Top-K from {old_top_k} "
                f"to {target_top_k}."
            )
            context.add_event("retrieval_top_k_changed")
            return

        plan.decision_trace.append(
            f"Unsupported feedback action: {action}"
        )
        context.add_event("invalid_feedback_action")

    def _sync_retriever_top_k(self, top_k):
        objects = [
            self.adaptive_retriever,
            getattr(
                self.adaptive_retriever,
                "dense_retriever",
                None
            ),
            getattr(
                self.adaptive_retriever,
                "bm25_retriever",
                None
            ),
            getattr(
                self.adaptive_retriever,
                "hybrid_retriever",
                None
            )
        ]

        seen = set()

        for retriever in objects:
            if retriever is None:
                continue

            identifier = id(retriever)
            if identifier in seen:
                continue
            seen.add(identifier)

            if hasattr(retriever, "top_k"):
                retriever.top_k = top_k

    @staticmethod
    def _retrieved_count(context):
        result = getattr(context, "retrieval_result", None)

        if result is None:
            return 0

        chunks = getattr(result, "retrieved_chunks", None)

        if chunks is None:
            return 0

        return len(chunks)

    @staticmethod
    def _planner_confidence_value(plan):
        results = plan.policy_results

        if not results:
            return 0.0

        values = [
            result.confidence
            for result in results.values()
        ]

        return sum(values) / len(values)
