from src.core.adaptive_context import AdaptiveContext
from src.planning.decision_types import RetrievalStrategy
from src.planning.policies.base_policy import BasePolicy
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.policy_result import PolicyResult


class RetrievalPolicy(BasePolicy):

    def apply(self, context: AdaptiveContext, plan: RetrievalPlan):

        query_type = context.query_analysis.get(
            "query_type",
            "semantic"
        )

        if query_type == "lexical":

            strategy = RetrievalStrategy.BM25
            confidence = 0.95
            reason = "Lexical query detected."

        elif query_type == "comparison":

            strategy = RetrievalStrategy.HYBRID
            confidence = 0.92
            reason = "Comparison query benefits from hybrid retrieval."

        else:

            strategy = RetrievalStrategy.DENSE
            confidence = 0.90
            reason = "Semantic query detected."

        plan.strategy = strategy

        plan.selected_policies.append(self.name)

        # plan.policy_confidence[self.name] = confidence
        # plan.policy_reasons[self.name] = reason
        
        plan.policy_results[self.name] = PolicyResult(

            policy_name=self.name,

            decision=f"strategy = {strategy.value}",

            confidence=confidence,

            reason=reason
        )

        plan.decision_trace.append(
            f"{self.name}: {strategy.value}"
        )

        return plan