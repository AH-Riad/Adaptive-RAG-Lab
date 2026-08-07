from src.core.adaptive_context import AdaptiveContext
from src.planning.policies.base_policy import BasePolicy
from src.planning.retrieval_plan import RetrievalPlan


class ChunkPolicy(BasePolicy):

    def apply(self, context, plan):

        query_type = context.query_analysis.get(
            "query_type",
            "semantic"
        )

        if query_type == "comparison":

            chunk = 400
            confidence = 0.90
            reason = "Comparison queries require larger context."

        elif query_type == "definition":

            chunk = 128
            confidence = 0.96
            reason = "Definition queries benefit from smaller chunks."

        else:

            chunk = 256
            confidence = 0.91
            reason = "Balanced default."

        plan.chunk_size = chunk

        plan.selected_policies.append(self.name)

        plan.policy_confidence[self.name] = confidence

        plan.policy_reasons[self.name] = reason

        plan.decision_trace.append(
            f"{self.name}: Chunk={chunk}"
        )

        return plan