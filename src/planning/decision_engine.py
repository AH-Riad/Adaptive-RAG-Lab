from __future__ import annotations
from statistics import mean

from src.core.adaptive_context import AdaptiveContext
from src.core.component import Component
from src.planning.decision_types import (
    PlannerConfidence,
    RetrievalDifficulty,
    RetrievalStrategy,
)
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.policy_registry import PolicyRegistry

class DecisionEngine(Component):
    """
    Creates a retrieval plan by applying independent planning policies.
    """

    def __init__(self):
        self.policies = PolicyRegistry.build()

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

        # 1. Run all policies
        for policy in self.policies:
            plan = policy.apply(context, plan)

        # 2. Calculate Global Confidence
        if plan.policy_results:
            confidences = [
                result.confidence
                for result in plan.policy_results.values()
            ]
            overall = mean(confidences)
        else:
            overall = 0.5 

        if overall >= 0.90:
            plan.planner_confidence = PlannerConfidence.HIGH
        elif overall >= 0.75:
            plan.planner_confidence = PlannerConfidence.MEDIUM
        else:
            plan.planner_confidence = PlannerConfidence.LOW

        plan.decision_trace.append(f"Overall Planner Confidence = {overall:.2f}")

        # 3. PHASE 4 UPDATE: GENERATE THE DECISION SCORE CARD
        report_lines = []
        report_lines.append("========================")
        report_lines.append("Planning Report")
        report_lines.append("========================")
        
        for name, result in plan.policy_results.items():
            report_lines.append(name.replace("Policy", "")) # e.g., RetrievalPolicy -> Retrieval
            report_lines.append(str(result.decision))
            report_lines.append("Confidence")
            report_lines.append(f"{result.confidence:.2f}")
            report_lines.append("Reason")
            report_lines.append(str(result.reason))
            report_lines.append("------------------------")
            
        report_lines.append("========================")
        report_lines.append("Overall Planning Score")
        report_lines.append(f"{overall:.2f}")
        
        score_card = "\n".join(report_lines)

        # Store both raw programmatic data AND the formatted string
        context.decision_report = {
            "overall_score": overall,
            "planner_confidence": plan.planner_confidence.value,
            "score_card": score_card,
            "policy_results": plan.policy_results,
        }

        # Finalize
        context.retrieval_plan = plan
        context.add_event("planning_completed")

        return context