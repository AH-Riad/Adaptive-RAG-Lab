from src.core.adaptive_context import AdaptiveContext
from src.planning.policies.base_policy import BasePolicy
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.policy_result import PolicyResult
from src.configs.config_loader import ConfigLoader  # <-- Added ConfigLoader


class TopKPolicy(BasePolicy):

    @property
    def version(self) -> str:
        return "1.1"

    @property
    def description(self) -> str:
        return "Determines Top-K retrieval count based on query complexity using YAML configuration."

    def apply(self, context, plan):
        # 1. Load the YAML configuration
        config = ConfigLoader.load("planning/topk_policy.yaml")
        
        # 2. Get complexity from the context
        complexity = context.query_analysis.get(
            "complexity",
            "medium"
        )

        # 3. Fetch settings from YAML (fallback to 'medium' if complexity is unknown)
        entry = config["complexity"].get(complexity, config["complexity"]["medium"])

        # 4. Extract values from the YAML entry
        top_k = entry["top_k"]
        confidence = entry["confidence"]
        reason = entry["reason"]

        # 5. Apply decisions to the plan
        plan.top_k = top_k
        plan.selected_policies.append(self.name)

        plan.policy_results[self.name] = PolicyResult(
            policy_name=self.name,
            decision=f"Top-K={top_k}",
            confidence=confidence,
            reason=reason
        )

        plan.decision_trace.append(
            f"{self.name}: Top-K={top_k} (v{self.version})"
        )

        return plan