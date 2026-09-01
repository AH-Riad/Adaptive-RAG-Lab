from src.core.component import Component
from src.adaptation.feedback_decision import (
    FeedbackDecision
)
from src.adaptation.calibrated_feedback_policy import (
    CalibratedFeedbackPolicy
)


class FeedbackController(Component):

    def __init__(
        self,
        policy_path: str,
        minimum_samples: int = 5
    ):

        self.policy = CalibratedFeedbackPolicy(
            policy_path=policy_path,
            minimum_samples=minimum_samples
        )

    @staticmethod
    def _confidence_bucket(
        confidence
    ):

        if confidence < 0.25:

            return "very_low"

        if confidence < 0.50:

            return "low"

        if confidence < 0.75:

            return "medium"

        return "high"

    def run(
        self,
        context
    ):

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

        query_analysis = (
            context.query_analysis
        )

        query_type = query_analysis.get(
            "query_type",
            "ambiguous"
        )

        current_strategy = (
            plan.strategy.value
        )

        current_top_k = (
            plan.top_k
        )

        confidence_bucket = (
            self._confidence_bucket(
                evidence.confidence
            )
        )

        strategy_result = (
            self.policy.get_strategy_action(
                query_type=query_type,
                current_strategy=current_strategy,
                confidence_bucket=confidence_bucket,
                top_k=current_top_k
            )
        )

        topk_result = (
            self.policy.get_topk_action(
                query_type=query_type,
                current_strategy=current_strategy,
                confidence_bucket=confidence_bucket,
                top_k=current_top_k
            )
        )

        selected_action = "keep"

        source = "fallback"

        reason = (
            "No sufficiently supported calibrated "
            "adaptive action was available."
        )

        target_strategy = None
        target_top_k = None

        strategy_utility = None
        topk_utility = None

        if strategy_result is not None:

            strategy_utility = (
                strategy_result[
                    "candidates"
                ].get(
                    strategy_result[
                        "selected_action"
                    ],
                    {}
                ).get(
                    "average_utility",
                    0.0
                )
            )

        if topk_result is not None:

            topk_utility = (
                topk_result[
                    "candidates"
                ].get(
                    topk_result[
                        "selected_action"
                    ],
                    {}
                ).get(
                    "average_utility",
                    0.0
                )
            )

        if (
            strategy_result is not None
            and
            topk_result is not None
        ):

            if (
                strategy_utility
                >=
                topk_utility
            ):

                selected_action = (
                    strategy_result[
                        "selected_action"
                    ]
                )

                source = (
                    "calibrated_strategy_policy"
                )

                reason = (
                    "Selected by the frozen "
                    "FiQA development strategy "
                    "action policy."
                )

            else:

                selected_action = (
                    topk_result[
                        "selected_action"
                    ]
                )

                source = (
                    "calibrated_topk_policy"
                )

                reason = (
                    "Selected by the frozen "
                    "FiQA development Top-K "
                    "action policy."
                )

        elif strategy_result is not None:

            selected_action = (
                strategy_result[
                    "selected_action"
                ]
            )

            source = (
                "calibrated_strategy_policy"
            )

            reason = (
                "Selected by the frozen "
                "FiQA development strategy "
                "action policy."
            )

        elif topk_result is not None:

            selected_action = (
                topk_result[
                    "selected_action"
                ]
            )

            source = (
                "calibrated_topk_policy"
            )

            reason = (
                "Selected by the frozen "
                "FiQA development Top-K "
                "action policy."
            )

        confidence = max(
            0.0,
            min(
                1.0,
                evidence.confidence
            )
        )

        if selected_action.startswith(
            "switch_to_"
        ):

            target_strategy = (
                selected_action[
                    len("switch_to_"):
                ]
            )

        elif selected_action.startswith(
            "set_top_k_"
        ):

            target_top_k = int(
                selected_action[
                    len("set_top_k_"):
                ]
            )

        decision = FeedbackDecision(
            action=selected_action,
            reason=reason,
            confidence=confidence,
            source=source,
            target_strategy=target_strategy,
            target_top_k=target_top_k
        )

        context.feedback_decision = decision

        context.add_event(
            "feedback_decision_created"
        )

        return context