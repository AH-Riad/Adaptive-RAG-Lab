from src.configs.config_loader import ConfigLoader
from src.planning.decision_types import RetrievalStrategy
from src.planning.policies.base_policy import BasePolicy
from src.planning.policy_result import PolicyResult


class RetrievalPolicy(BasePolicy):

    def __init__(self):

        self.config = ConfigLoader.load(
            "planning/retrieval_policy.yaml"
        )

    def apply(self, context, plan):

        query_type = context.query_analysis.get(
            "query_type",
            "semantic"
        )

        query_config = self.config.get(
            "query_types",
            {}
        )

        entry = query_config.get(
            query_type
        )

        if entry is None:

            entry = query_config.get(
                "semantic"
            )

        strategy = RetrievalStrategy(
            entry["strategy"]
        )

        confidence = float(
            entry["confidence"]
        )

        reason = str(
            entry["reason"]
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
                f"strategy = {strategy.value}"
            ),
            confidence=confidence,
            reason=reason
        )

        plan.decision_trace.append(
            f"{self.name}: "
            f"{strategy.value} "
            f"(v{self.version})"
        )

        return plan

    @property
    def version(self):

        return "1.2"

    @property
    def description(self):

        return (
            "Selects Dense, BM25, or Hybrid "
            "retrieval from query type."
        )