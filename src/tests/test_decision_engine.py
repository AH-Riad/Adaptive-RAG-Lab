from src.core.adaptive_context import AdaptiveContext
from src.planning.decision_engine import DecisionEngine


def main():

    context = AdaptiveContext(

        query="Compare BERT and GPT"

    )

    context.query_analysis = {

        "query_type": "comparison",

        "complexity": "high",

    }

    engine = DecisionEngine()

    context = engine.run(context)

    plan = context.retrieval_plan

    print("=" * 70)

    print("Retrieval Strategy")

    print(plan.strategy.value)

    print("=" * 70)

    print("Top-K")

    print(plan.top_k)

    print("=" * 70)

    print("Chunk Size")

    print(plan.chunk_size)

    print("=" * 70)

    print("Planner Confidence")

    print(plan.planner_confidence.value)

    print("=" * 70)

    print("Decision Trace")

    for item in plan.decision_trace:

        print(item)

    print("=" * 70)

    print("Policy Results")

    for result in plan.policy_results.values():

        print(result)


if __name__ == "__main__":
    main()