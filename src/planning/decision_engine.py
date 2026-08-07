from __future__ import annotations

from statistics import mean

from src.core.adaptive_context import AdaptiveContext
from src.core.component import Component
from src.planning.decision_types import (
    PlannerConfidence,
    RetrievalDifficulty,
    RetrievalStrategy,
)
from src.planning.policies.chunk_policy import ChunkPolicy
from src.planning.policies.retrieval_policy import RetrievalPolicy
from src.planning.policies.topk_policy import TopKPolicy
from src.planning.retrieval_plan import RetrievalPlan


class DecisionEngine(Component):
    """
    Creates a retrieval plan by applying
    independent planning policies.
    """

    def __init__(self):

        self.policies = [

            RetrievalPolicy(),

            TopKPolicy(),

            ChunkPolicy(),

        ]

    def run(
        self,
        context: AdaptiveContext,
    ) -> AdaptiveContext:

        plan = RetrievalPlan(

            strategy=RetrievalStrategy.DENSE,

            top_k=5,

            chunk_size=256,

            chunk_overlap=30,

            difficulty=RetrievalDifficulty.MEDIUM,

            planner_confidence=PlannerConfidence.MEDIUM,

        )

        for policy in self.policies:

            plan = policy.apply(context, plan)

        confidences = [

            result.confidence

            for result in plan.policy_results.values()

        ]

        overall = mean(confidences)

        if overall >= 0.90:

            plan.planner_confidence = PlannerConfidence.HIGH

        elif overall >= 0.75:

            plan.planner_confidence = PlannerConfidence.MEDIUM

        else:

            plan.planner_confidence = PlannerConfidence.LOW

        plan.decision_trace.append(

            f"Overall Planner Confidence = {overall:.2f}"

        )

        context.retrieval_plan = plan

        context.add_event("planning_completed")

        return context