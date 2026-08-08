from src.planning.policies.base_policy import BasePolicy
from src.planning.policy_result import PolicyResult


class ConfidencePolicy(BasePolicy):

    def apply(self, context, plan):

        score = context.query_analysis.get(
            "query_type_confidence",
            1.0
        )

        if score < 0.60:

            plan.rewrite_query = True

            reason = (
                "Low analysis confidence. "
                "Enable query rewriting."
            )

        else:

            reason = (
                "Analysis confidence acceptable."
            )

        plan.selected_policies.append(self.name)

        plan.policy_results[self.name] = PolicyResult(

            policy_name=self.name,

            decision=f"Rewrite={plan.rewrite_query}",

            confidence=score,

            reason=reason
        )

        plan.decision_trace.append(

            f"{self.name}: Rewrite={plan.rewrite_query}"

        )

        return plan