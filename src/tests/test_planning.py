from src.planning.decision_types import (
    PlannerConfidence,
    RetrievalDifficulty,
    RetrievalStrategy,
)
from src.planning.retrieval_plan import RetrievalPlan


def main():

    plan = RetrievalPlan(
        strategy=RetrievalStrategy.DENSE,
        top_k=5,
        chunk_size=150,
        chunk_overlap=30,
        rerank=False,
        rewrite_query=False,
        difficulty=RetrievalDifficulty.MEDIUM,
        planner_confidence=PlannerConfidence.HIGH,
        decision_trace=["Initial dense retrieval."]
    )

    print("=" * 60)

    print(plan)

    print("=" * 60)

    print(plan.strategy.value)

    print(plan.planner_confidence.value)


if __name__ == "__main__":
    main()