from src.core.adaptive_context import AdaptiveContext
from src.planning.policies.base_policy import BasePolicy
from src.planning.retrieval_plan import RetrievalPlan


class TopKPolicy(BasePolicy):

    def apply(self, context, plan):

        complexity = context.query_analysis.get(
            "complexity",
            "medium"
        )

        if complexity == "low":

            top_k = 3
            confidence = 0.95
            reason = "Simple query."

        elif complexity == "high":

            top_k = 8
            confidence = 0.90
            reason = "Complex query."

        else:

            top_k = 5
            confidence = 0.92
            reason = "Moderate complexity."

        plan.top_k = top_k

        plan.selected_policies.append(self.name)

        plan.policy_confidence[self.name] = confidence

        plan.policy_reasons[self.name] = reason

        plan.decision_trace.append(
            f"{self.name}: Top-K={top_k}"
        )

        return plan