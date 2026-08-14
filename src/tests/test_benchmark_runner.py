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

from src.evaluation.evaluation_dataset import (
    build_ground_truth
)

from src.evaluation.benchmark_runner import (
    BenchmarkRunner
)


def build_systems():

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
        "\nInitializing embeddings..."
    )

    embedding_model = (
        SentenceTransformerEmbedding()
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

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5
    )

    bm25 = BM25Retriever(
        chunks=chunks,
        top_k=5
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        top_k=5,
        alpha=0.7
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        hybrid_retriever=hybrid
    )

    orchestrator = (
        AdaptiveRetrievalOrchestrator(
            adaptive_retriever=adaptive_retriever,
            max_retries=2
        )
    )

    d2rag = D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=orchestrator
    )

    return (
        {
            "Dense": dense,
            "BM25": bm25,
            "Hybrid": hybrid
        },
        d2rag
    )


def main():

    print("=" * 60)
    print("D²RAG BENCHMARK")
    print("=" * 60)

    (
        baselines,
        d2rag
    ) = build_systems()

    ground_truth = (
        build_ground_truth()
    )

    runner = BenchmarkRunner(
        ground_truth=ground_truth
    )

    print(
        "\nRunning fixed baselines..."
    )

    baseline_results = (
        runner.run_baselines(
            baselines
        )
    )

    print(
        "Running D²RAG..."
    )

    d2rag_results = (
        runner.run_d2rag(
            d2rag
        )
    )

    results = (
        baseline_results
        +
        d2rag_results
    )

    summary = runner.aggregate(
        results
    )

    print("\n" + "=" * 60)
    print("AVERAGE RESULTS")
    print("=" * 60)

    for system, values in (
        summary.items()
    ):

        print(
            f"\n{system}"
        )

        print(
            "Precision@5:",
            round(
                values["precision_at_5"],
                4
            )
        )

        print(
            "Recall@5:",
            round(
                values["recall_at_5"],
                4
            )
        )

        print(
            "MRR:",
            round(
                values["mrr"],
                4
            )
        )

        print(
            "nDCG@5:",
            round(
                values["ndcg_at_5"],
                4
            )
        )

    jsonl_path, csv_path = (
        runner.save_results(
            results
        )
    )

    print("\n" + "=" * 60)

    print(
        "JSONL:",
        jsonl_path
    )

    print(
        "CSV:",
        csv_path
    )

    print("=" * 60)


if __name__ == "__main__":
    main()