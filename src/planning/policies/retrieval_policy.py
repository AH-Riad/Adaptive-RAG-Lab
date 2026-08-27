from src.planning.decision_types import (
    RetrievalStrategy
)

from src.planning.policies.base_policy import (
    BasePolicy
)

from src.planning.policy_result import (
    PolicyResult
)

from src.planning.calibrated_policy import (
    CalibratedPolicy
)


class RetrievalPolicy(BasePolicy):

    def __init__(self):

        self.calibrated_policy = (
            CalibratedPolicy(
                path=(
                    "results/logs/"
                    "fiqa_dev_strategy_policy_v1.json"
                )
            )
        )

    def apply(
        self,
        context,
        plan
    ):

        query_type = context.query_analysis.get(
            "query_type",
            "ambiguous"
        )

        strategy = (
            self.calibrated_policy.get_strategy(
                query_type
            )
        )

        confidence = 0.95

        reason = (
            "Strategy selected from the "
            "frozen FiQA development calibration "
            f"policy v{self.calibrated_policy.version}."
        )

        plan.strategy = strategy

        plan.selected_policies.append(
            self.name
        )

        plan.policy_results[
            self.name
        ] = PolicyResult(
            policy_name=self.name,
            decision=(
                f"strategy = "
                f"{strategy.value}"
            ),
            confidence=confidence,
            reason=reason
        )

        plan.decision_trace.append(
            f"{self.name}: "
            f"{strategy.value} "
            f"(calibrated policy "
            f"v{self.calibrated_policy.version})"
        )

        return plan

    @property
    def version(self):

        return "2.0"

    @property
    def description(self):

        return (
            "Selects retrieval strategy using "
            "a frozen development-set "
            "calibration policy."
        )