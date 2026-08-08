from src.core.adaptive_context import AdaptiveContext
from src.planning.policies.base_policy import BasePolicy
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.policy_result import PolicyResult
from src.configs.config_loader import ConfigLoader  # <-- Added ConfigLoader


class ChunkPolicy(BasePolicy):

    @property
    def version(self) -> str:
        return "1.1"

    @property
    def description(self) -> str:
        return "Determines chunk size based on query type using YAML configuration."

    def apply(self, context, plan):
        # 1. Load the YAML configuration
        config = ConfigLoader.load("planning/chunk_policy.yaml")
        
        # 2. Get the query type from the context
        query_type = context.query_analysis.get(
            "query_type",
            "semantic"
        )

        # 3. Fetch settings from YAML (fallback to 'semantic' if query_type is unknown)
        entry = config["query_types"].get(query_type, config["query_types"]["semantic"])

        # 4. Extract values from the YAML entry
        chunk = entry["chunk_size"]
        confidence = entry["confidence"]
        reason = entry["reason"]

        # 5. Apply decisions to the plan
        plan.chunk_size = chunk
        plan.selected_policies.append(self.name)

        plan.policy_results[self.name] = PolicyResult(
            policy_name=self.name,
            decision=f"Chunk={chunk}",
            confidence=confidence,
            reason=reason
        )
                
        plan.decision_trace.append(
            f"{self.name}: Chunk={chunk} (v{self.version})"
        )

        return plan