from src.core.adaptive_context import AdaptiveContext
from src.assessment.evidence_assessor import EvidenceAssessor
from src.retrievers.retrieval_result import RetrievalResult
from src.core import RetrievedChunk


def main():

    print("=" * 60)
    print("EVIDENCE ASSESSMENT TEST")
    print("=" * 60)

    chunks = [

        RetrievedChunk(
            chunk_id="chunk_001",
            text="Transformer uses self-attention.",
            score=0.6759,
            metadata={}
        ),

        RetrievedChunk(
            chunk_id="chunk_003",
            text="Multi-head attention uses multiple representations.",
            score=0.6739,
            metadata={}
        ),

        RetrievedChunk(
            chunk_id="chunk_004",
            text="Positional encoding provides token order.",
            score=0.5591,
            metadata={}
        ),
    ]

    retrieval_result = RetrievalResult(
        query="Explain self-attention",
        retrieved_chunks=chunks
    )

    context = AdaptiveContext(
        query="Explain self-attention"
    )

    context.retrieval_result = retrieval_result

    assessor = EvidenceAssessor()

    context = assessor.run(context)

    result = context.evidence_result

    print("\nRetrieved Count :", result.retrieved_count)

    print("Relevant Count  :", result.relevant_count)

    print(
        "Average Score   :",
        round(result.average_score, 4)
    )

    print(
        "Coverage        :",
        round(result.coverage, 4)
    )

    print(
        "Confidence      :",
        round(result.confidence, 4)
    )

    print(
        "Threshold       :",
        result.threshold
    )

    print(
        "Accepted        :",
        result.accepted
    )

    print("\nReasons:")

    for reason in result.reasons:
        print("-", reason)

    print("\nRecommendations:")

    for recommendation in result.recommendations:
        print("-", recommendation)

    print("\n" + "=" * 60)

    assert result.retrieved_count == 3

    assert result.relevant_count == 2

    assert 0.0 <= result.coverage <= 1.0

    assert 0.0 <= result.confidence <= 1.0

    assert result.accepted is True

    print("EVIDENCE ASSESSOR TEST PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()