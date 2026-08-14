from src.core.adaptive_context import AdaptiveContext
from src.assessment.evidence_assessor import EvidenceAssessor
from src.retrievers.retrieval_result import RetrievalResult
from src.core import RetrievedChunk

def main():
    print("EVIDENCE ASSESSMENT TEST\n")

    # Test 1
    chunks = [
        RetrievedChunk(chunk_id="chunk_001", text="Transformer uses self-attention.", score=0.6759, metadata={}),
        RetrievedChunk(chunk_id="chunk_003", text="Multi-head attention uses multiple representations.", score=0.6739, metadata={}),
        RetrievedChunk(chunk_id="chunk_004", text="Positional encoding provides token order.", score=0.5591, metadata={}),
    ]
    retrieval_result = RetrievalResult(query="Explain self-attention", retrieved_chunks=chunks)
    context = AdaptiveContext(query="Explain self-attention")
    context.query_analysis = {"query_type": "semantic"}
    context.retrieval_result = retrieval_result
    assessor = EvidenceAssessor()
    context = assessor.run(context)
    
    print("[Test 1] Standard Assessment:")
    print("Retrieved Count :", context.evidence_result.retrieved_count)
    print("Relevant Count  :", context.evidence_result.relevant_count)
    print("Average Score   :", round(context.evidence_result.average_score, 4))
    print("Coverage        :", round(context.evidence_result.coverage, 4))
    print("Confidence      :", round(context.evidence_result.confidence, 4))
    print("Threshold       :", context.evidence_result.threshold)
    print("Accepted        :", context.evidence_result.accepted)
    
    # Test 2
    print("\n[Test 2] Single-evidence lexical query")
    lexical_context = AdaptiveContext(query="query key value representations")
    lexical_context.query_analysis = {"query_type": "lexical"}
    
    # Adjusted the score of other_1 to 0.40 so the average brings the confidence above 0.55
    lexical_context.retrieval_result = RetrievalResult(
        query="query key value representations",
        retrieved_chunks=[
            RetrievedChunk(chunk_id="sample_retrieval_corpus_CHUNK_004", text="query, key, and value representations", score=0.95, metadata={}),
            RetrievedChunk(chunk_id="other_1", text="unrelated information", score=0.40, metadata={})
        ]
    )
    
    lexical_context = assessor.run(lexical_context)
    result2 = lexical_context.evidence_result

    print("Retrieved Count :", result2.retrieved_count)
    print("Relevant Count  :", result2.relevant_count)
    print("Average Score   :", round(result2.average_score, 4))
    print("Coverage        :", round(result2.coverage, 4))
    print("Confidence      :", round(result2.confidence, 4))
    print("Threshold       :", result2.threshold)
    print("Accepted        :", result2.accepted)
    
    print("\nReasons:")
    for reason in result2.reasons:
        print("-", reason)
        
    print("\nRecommendations:")
    for rec in result2.recommendations:
        print("-", rec)

    assert result2.accepted is True
    print("\nEVIDENCE ASSESSOR TEST FULLY PASSED")

if __name__ == "__main__":
    main()