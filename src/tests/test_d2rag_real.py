from src.core.adaptive_context import AdaptiveContext

from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from src.vectorstore.chroma_store import ChromaVectorStore

from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.adaptive_retriever import AdaptiveRetriever

from src.analyzer.query_analyzer import QueryAnalyzer

from src.adaptation.feedback_controller import (
    FeedbackController
)

from src.adaptation.adaptive_retrieval_orchestrator import (
    AdaptiveRetrievalOrchestrator
)

from src.adaptation.d2rag_engine import D2RAGEngine


def main():

    print("=" * 60)
    print("REAL D²RAG TEST")
    print("=" * 60)

    loader = TextLoader()

    documents = loader.load(
        "datasets/sample_retrieval_corpus.txt"
    )

    print(
        f"Loaded Documents: {len(documents)}"
    )

    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(
        documents
    )

    print(
        f"Created Chunks: {len(chunks)}"
    )

    print(
        "\nInitializing embedding model..."
    )

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    print(
        "Creating embeddings..."
    )

    embeddings = embedding_model.encode(
        chunks
    )

    print(
        "Initializing vector store..."
    )

    vector_store = ChromaVectorStore()

    vector_store.reset()

    vector_store.add(
        embeddings
    )

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks,
        top_k=5
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        top_k=5,
        alpha=0.7
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever
    )

    orchestrator = AdaptiveRetrievalOrchestrator(
        adaptive_retriever=adaptive_retriever,
        max_retries=2
    )

    engine = D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=orchestrator
    )

    queries = [
        "self-attention",
        "How does the Transformer process information?",
        "How are Transformers different from recurrent neural networks?",
        "query key value representations",
        "attention"
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print("QUERY")
        print("=" * 60)

        print(query)

        context = AdaptiveContext(
            query=query
        )

        context = engine.run(
            context
        )

        plan = context.retrieval_plan
        evidence = context.evidence_result

        print(
            "\nQuery Type:",
            context.query_analysis[
                "query_type"
            ]
        )

        print(
            "Planner Confidence:",
            plan.planner_confidence.value
        )

        print(
            "Selected Strategy:",
            plan.strategy.value
        )

        print(
            "Top-K:",
            plan.top_k
        )

        print(
            "Evidence Confidence:",
            round(
                evidence.confidence,
                4
            )
        )

        print(
            "Evidence Accepted:",
            evidence.accepted
        )

        print(
            "Attempts:",
            context.decision_report[
                "retrieval_attempts"
            ]
        )

        print(
            "Status:",
            context.decision_report[
                "adaptive_retrieval_status"
            ]
        )

        print(
            "Retrieved Chunks:"
        )

        for chunk in (
            context.retrieval_result.retrieved_chunks
        ):

            print(
                chunk.chunk_id,
                "->",
                round(chunk.score, 4)
            )

    print("\n" + "=" * 60)
    print("REAL D²RAG TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()