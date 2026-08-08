from src.core.adaptive_context import AdaptiveContext
from src.planning.decision_types import RetrievalStrategy
from src.planning.policies.base_policy import BasePolicy
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.policy_result import PolicyResult
from src.configs.config_loader import ConfigLoader  

class RetrievalPolicy(BasePolicy):

    @property
    def version(self) -> str:
        return "1.1"

    @property
    def description(self) -> str:
        return "Selects retrieval strategy based on YAML configuration mapping."

    def apply(self, context: AdaptiveContext, plan: RetrievalPlan):
        
        # 1. Load the configuration file
        config = ConfigLoader.load("planning/retrieval_policy.yaml")
        
        # 2. Get the query type from the context
        query_type = context.query_analysis.get("query_type", "semantic")

        # 3. Fetch the exact settings from the YAML file (fallback to 'semantic' if unknown)
        entry = config["query_types"].get(query_type, config["query_types"]["semantic"])

        # 4. Extract values from YAML
        strategy = RetrievalStrategy(entry["strategy"])  # Converts string "hybrid" to Enum
        confidence = entry["confidence"]
        reason = entry["reason"]

        plan.strategy = strategy
        plan.selected_policies.append(self.name)

        plan.policy_results[self.name] = PolicyResult(
            policy_name=self.name,
            decision=f"strategy = {strategy.value}",
            confidence=confidence,
            reason=reason
        )

        plan.decision_trace.append(f"{self.name}: {strategy.value} (v{self.version})")

        return plan