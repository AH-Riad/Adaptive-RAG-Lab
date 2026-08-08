from src.core.adaptive_context import AdaptiveContext
from src.adaptation.adaptive_retrieval_orchestrator import (
    AdaptiveRetrievalOrchestrator,
)
from src.assessment.evidence_assessor import EvidenceAssessor
from src.adaptation.feedback_controller import FeedbackController
from src.planning.decision_engine import DecisionEngine


class MockRetriever:

    def __init__(self):

        self.top_k = 3

        self.call_count = 0

    def retrieve(self, query):

        self.call_count += 1

        # First attempt intentionally produces
        # weak evidence.
        if self.call_count == 1:

            from src.core import RetrievedChunk
            from src.retrievers.retrieval_result import RetrievalResult

            chunks = [
                RetrievedChunk(
                    chunk_id="weak_1",
                    text="Irrelevant information.",
                    score=0.30,
                    metadata={},
                ),
                RetrievedChunk(
                    chunk_id="weak_2",
                    text="Unrelated information.",
                    score=0.35,
                    metadata={},
                ),
                RetrievedChunk(
                    chunk_id="weak_3",
                    text="Another unrelated chunk.",
                    score=0.40,
                    metadata={},
                ),
            ]

        # Second attempt produces strong evidence.
        else:

            from src.core import RetrievedChunk
            from src.retrievers.retrieval_result import RetrievalResult

            chunks = [
                RetrievedChunk(
                    chunk_id="good_1",
                    text="The Transformer relies on self-attention.",
                    score=0.85,
                    metadata={},
                ),
                RetrievedChunk(
                    chunk_id="good_2",
                    text="Multi-head attention uses multiple representations.",
                    score=0.80,
                    metadata={},
                ),
                RetrievedChunk(
                    chunk_id="good_3",
                    text="Self-attention allows tokens to interact.",
                    score=0.78,
                    metadata={},
                ),
            ]

        return RetrievalResult(
            query=query,
            retrieved_chunks=chunks,
        )


def main():

    print("=" * 60)
    print("ADAPTIVE RETRIEVAL INTEGRATION TEST")
    print("=" * 60)

    context = AdaptiveContext(
        query="Explain self-attention"
    )

    retriever = MockRetriever()

    orchestrator = AdaptiveRetrievalOrchestrator(
        retriever=retriever,
        decision_engine=DecisionEngine(),
        evidence_assessor=EvidenceAssessor(),
        feedback_controller=FeedbackController(),
        max_retries=2,
    )

    context = orchestrator.run(context)

    print("\nFinal Status:")
    print(
        context.decision_report[
            "adaptive_retrieval_status"
        ]
    )

    print(
        "\nRetrieval Attempts:",
        context.decision_report[
            "retrieval_attempts"
        ],
    )

    print(
        "\nFinal Top-K:",
        context.retrieval_plan.top_k,
    )

    print(
        "\nFinal Evidence Confidence:",
        round(
            context.evidence_result.confidence,
            4,
        ),
    )

    print("\nEvents:")

    for event in context.events:
        print("-", event)

    print("\nDecision Trace:")

    for item in context.retrieval_plan.decision_trace:
        print("-", item)

    print("\n" + "=" * 60)

    assert retriever.call_count == 2

    assert (
        context.decision_report[
            "adaptive_retrieval_status"
        ]
        == "accepted"
    )

    assert (
        context.decision_report[
            "retrieval_attempts"
        ]
        == 2
    )

    assert context.evidence_result.accepted is True

    print("ADAPTIVE RETRIEVAL TEST PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()