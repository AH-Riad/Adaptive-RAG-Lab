from src.core.adaptive_context import (
    AdaptiveContext
)

from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import (
    RecursiveChunker
)

from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from src.vectorstore.chroma_store import (
    ChromaVectorStore
)

from src.retrievers.dense_retriever import (
    DenseRetriever
)

from src.retrievers.bm25_retriever import (
    BM25Retriever
)

from src.retrievers.hybrid_retriever import (
    HybridRetriever
)

from src.retrievers.adaptive_retriever import (
    AdaptiveRetriever
)

from src.analyzer.query_analyzer import (
    QueryAnalyzer
)

from src.adaptation.adaptive_retrieval_orchestrator import (
    AdaptiveRetrievalOrchestrator
)

from src.adaptation.d2rag_engine import (
    D2RAGEngine
)


def main():

    print("=" * 60)
    print("D²RAG STRATEGY TRACKER TEST")
    print("=" * 60)

    loader = TextLoader()

    documents = loader.load(
        "datasets/sample_retrieval_corpus.txt"
    )

    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(
        documents
    )

    print(
        f"Loaded Documents: {len(documents)}"
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

        context = AdaptiveContext(
            query=query
        )

        context = engine.run(
            context
        )

        report = context.decision_report

        print("\n" + "=" * 60)
        print("QUERY")
        print("=" * 60)

        print(
            query
        )

        print(
            "\nQuery Type:",
            context.query_analysis[
                "query_type"
            ]
        )

        print(
            "Initial Strategy:",
            report[
                "initial_strategy"
            ]
        )

        print(
            "Initial Top-K:",
            report[
                "initial_top_k"
            ]
        )

        print(
            "Planner Confidence:",
            round(
                report[
                    "initial_planner_confidence"
                ],
                4
            )
        )

        print(
            "\nAttempt History:"
        )

        for attempt in report[
            "attempt_history"
        ]:

            print(
                f"Attempt "
                f"{attempt['attempt_number']}: "
                f"strategy="
                f"{attempt['strategy']}, "
                f"top_k="
                f"{attempt['top_k']}, "
                f"evidence="
                f"{attempt['evidence_confidence']:.4f}, "
                f"accepted="
                f"{attempt['evidence_accepted']}"
            )

        print(
            "\nStrategy Transitions:"
        )

        transitions = report[
            "strategy_transitions"
        ]

        if transitions:

            for transition in transitions:

                print(
                    f"Attempt "
                    f"{transition['attempt_number']}: "
                    f"{transition['old_strategy']} "
                    f"-> "
                    f"{transition['new_strategy']}"
                )

        else:

            print(
                "No strategy transition."
            )

        print(
            "\nFinal Strategy:",
            report[
                "final_strategy"
            ]
        )

        print(
            "Final Top-K:",
            report[
                "final_top_k"
            ]
        )

        print(
            "Final Evidence Confidence:",
            round(
                report[
                    "final_evidence_confidence"
                ],
                4
            )
        )

        print(
            "Attempts:",
            report[
                "retrieval_attempts"
            ]
        )

        print(
            "Status:",
            report[
                "adaptive_retrieval_status"
            ]
        )

    print("\n" + "=" * 60)
    print("D²RAG STRATEGY TRACKER TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()