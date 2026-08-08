from src.core.adaptive_context import AdaptiveContext
from src.adaptation.feedback_controller import FeedbackController
from src.assessment.evidence_result import EvidenceResult
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.decision_types import (
    RetrievalStrategy,
    RetrievalDifficulty,
    PlannerConfidence,
)


def main():

    print("=" * 60)
    print("FEEDBACK CONTROLLER TEST")
    print("=" * 60)

    context = AdaptiveContext(
        query="Explain self-attention"
    )

    context.retrieval_plan = RetrievalPlan(
        strategy=RetrievalStrategy.DENSE,
        top_k=5,
        chunk_size=150,
        chunk_overlap=30,
        difficulty=RetrievalDifficulty.MEDIUM,
        planner_confidence=PlannerConfidence.HIGH,
    )

    context.evidence_result = EvidenceResult(
        accepted=False,
        # Good evidence test
        # confidence=0.82,
        # average_score=0.78,
        # coverage=0.80,
        
        # Bad evidence test
        confidence=0.35,
        average_score=0.42,
        coverage=0.20,
        
        retrieved_count=5,
        relevant_count=4,
        threshold=0.65,
    )

    controller = FeedbackController()

    context = controller.run(context)

    decision = context.feedback_decision

    print("\nShould Retry :", decision.should_retry)
    print("New Top-K    :", decision.new_top_k)
    print(
        "Change Strategy :",
        decision.change_strategy
    )
    print(
        "New Strategy :",
        decision.new_strategy
    )
    print(
        "Rewrite Query :",
        decision.rewrite_query
    )
    print(
        "Rerank       :",
        decision.rerank
    )
    print(
        "Confidence   :",
        round(decision.confidence, 4)
    )

    print("\nActions:")

    for action in decision.actions:
        print("-", action)

    assert decision.should_retry is True
    assert decision.new_top_k == 10
    assert decision.change_strategy is True

    print("\n" + "=" * 60)
    print("BAD EVIDENCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()