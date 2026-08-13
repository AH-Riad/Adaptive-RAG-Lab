from src.core.component import Component
from src.adaptation.feedback_decision import FeedbackDecision
from src.planning.decision_types import RetrievalStrategy


class FeedbackController(Component):

    def __init__(
        self,
        retry_threshold: float = 0.55,
        max_top_k: int = 10
    ):
        self.retry_threshold = retry_threshold
        self.max_top_k = max_top_k

    def run(self, context):

        evidence = context.evidence_result
        plan = context.retrieval_plan

        if evidence is None:
            raise RuntimeError(
                "FeedbackController requires "
                "evidence_result."
            )

        if plan is None:
            raise RuntimeError(
                "FeedbackController requires "
                "retrieval_plan."
            )

        if evidence.accepted:

            decision = FeedbackDecision(
                should_retry=False,
                new_top_k=plan.top_k,
                change_strategy=False,
                new_strategy=None,
                rewrite_query=False,
                rerank=False,
                reason=(
                    "Evidence is sufficient."
                ),
                confidence=evidence.confidence,
                actions=[
                    "accept_retrieval"
                ]
            )

            context.feedback_decision = decision

            context.add_event(
                "feedback_evaluation_completed"
            )

            return context

        new_top_k = min(
            plan.top_k * 2,
            self.max_top_k
        )

        actions = [
            f"increase_top_k:{plan.top_k}->{new_top_k}"
        ]

        change_strategy = False
        new_strategy = None

        if evidence.coverage < 0.40:

            change_strategy = True

            if plan.strategy == RetrievalStrategy.DENSE:

                new_strategy = (
                    RetrievalStrategy.HYBRID.value
                )

            elif plan.strategy == RetrievalStrategy.BM25:

                new_strategy = (
                    RetrievalStrategy.HYBRID.value
                )

            elif plan.strategy == RetrievalStrategy.HYBRID:

                new_strategy = (
                    RetrievalStrategy.DENSE.value
                )

            actions.append(
                f"change_strategy:{new_strategy}"
            )

        rewrite_query = (
            evidence.average_score < 0.45
        )

        if rewrite_query:

            actions.append(
                "rewrite_query"
            )

        rerank = (
            evidence.retrieved_count >= 5
            and
            evidence.average_score >= 0.45
        )

        if rerank:

            actions.append(
                "rerank_results"
            )

        decision = FeedbackDecision(
            should_retry=True,
            new_top_k=new_top_k,
            change_strategy=change_strategy,
            new_strategy=new_strategy,
            rewrite_query=rewrite_query,
            rerank=rerank,
            reason=(
                "Evidence did not meet the "
                "acceptance criteria."
            ),
            confidence=evidence.confidence,
            actions=actions
        )

        context.feedback_decision = decision

        context.add_event(
            "feedback_evaluation_completed"
        )

        return context