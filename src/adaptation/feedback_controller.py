from src.core.component import Component
from src.adaptation.feedback_decision import FeedbackDecision
from src.planning.decision_types import RetrievalStrategy


class FeedbackController(Component):
    """
    Determines how the retrieval process should adapt
    when retrieved evidence is insufficient.
    """

    def __init__(
        self,
        retry_threshold: float = 0.65,
        max_top_k: int = 10,
    ):
        self.retry_threshold = retry_threshold
        self.max_top_k = max_top_k

    def run(self, context):

        evidence = context.evidence_result
        plan = context.retrieval_plan

        if evidence is None:
            raise RuntimeError(
                "FeedbackController requires evidence_result."
            )

        if plan is None:
            raise RuntimeError(
                "FeedbackController requires retrieval_plan."
            )

        # --------------------------------------------------
        # CASE 1: Evidence is already good
        # --------------------------------------------------

        if evidence.confidence >= self.retry_threshold:

            decision = FeedbackDecision(
                should_retry=False,
                new_top_k=plan.top_k,
                change_strategy=False,
                new_strategy=None,
                rewrite_query=False,
                rerank=False,
                reason=(
                    "Evidence confidence is sufficient. "
                    "No retrieval adaptation is required."
                ),
                confidence=evidence.confidence,
                actions=[
                    "accept_retrieval"
                ],
            )

            context.feedback_decision = decision

            context.add_event(
                "feedback_evaluation_completed"
            )

            return context

        # --------------------------------------------------
        # CASE 2: Evidence is weak
        # --------------------------------------------------

        actions = []

        new_top_k = min(
            plan.top_k * 2,
            self.max_top_k
        )

        actions.append(
            f"increase_top_k:{plan.top_k}->{new_top_k}"
        )

        # --------------------------------------------------
        # Determine whether to change strategy
        # --------------------------------------------------

        change_strategy = False
        new_strategy = None

        if evidence.coverage < 0.50:

            change_strategy = True

            if plan.strategy == RetrievalStrategy.DENSE:

                new_strategy = (
                    RetrievalStrategy.HYBRID.value
                )

            elif plan.strategy == RetrievalStrategy.BM25:

                new_strategy = (
                    RetrievalStrategy.HYBRID.value
                )

            else:

                new_strategy = (
                    RetrievalStrategy.DENSE.value
                )

            actions.append(
                f"change_strategy:{new_strategy}"
            )

        # --------------------------------------------------
        # Query rewriting
        # --------------------------------------------------

        rewrite_query = False

        if evidence.average_score < 0.60:

            rewrite_query = True

            actions.append(
                "rewrite_query"
            )

        # --------------------------------------------------
        # Reranking
        # --------------------------------------------------

        rerank = False

        if (
            evidence.retrieved_count >= 5
            and evidence.average_score >= 0.50
        ):

            rerank = True

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
                "Evidence confidence is below the retry "
                "threshold. Retrieval adaptation is required."
            ),
            confidence=evidence.confidence,
            actions=actions,
        )

        context.feedback_decision = decision

        context.add_event(
            "feedback_evaluation_completed"
        )

        return context