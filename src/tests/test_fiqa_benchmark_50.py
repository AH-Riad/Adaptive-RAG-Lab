from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.dense_benchmark_index import DenseBenchmarkIndex
from src.evaluation.bm25s_benchmark_index import BM25SBenchmarkIndex

from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from src.retrievers.benchmark_dense_retriever import (
    BenchmarkDenseRetriever
)

from src.retrievers.benchmark_bm25s_retriever import (
    BenchmarkBM25SRetriever
)

from src.retrievers.benchmark_hybrid_retriever import (
    BenchmarkHybridRetriever
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

from src.evaluation.beir_benchmark_runner import (
    BEIRBenchmarkRunner
)


def build_systems(
    documents_by_id,
    dense_index,
    bm25s_index,
    embedding_model
):

    dense = BenchmarkDenseRetriever(
        index=dense_index,
        documents_by_id=documents_by_id,
        embedding_model=embedding_model,
        top_k=5
    )

    bm25s = BenchmarkBM25SRetriever(
        index=bm25s_index,
        documents_by_id=documents_by_id,
        top_k=5
    )

    hybrid = BenchmarkHybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25s,
        top_k=5,
        alpha=0.7
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25s,
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

    return {
        "Dense": dense,
        "BM25S": bm25s,
        "Hybrid": hybrid
    }, d2rag


def main():

    print("=" * 60)
    print("FIQA 50-QUERY CONTROLLED BENCHMARK")
    print("=" * 60)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = (
        dataset.load(
            split="test"
        )
    )

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus
    )

    documents = (
        benchmark_corpus.to_documents()
    )

    documents_by_id = {
        document.id: document
        for document in documents
    }

    print(
        "Corpus:",
        len(documents)
    )

    print(
        "Total queries:",
        len(queries)
    )

    dense_index = DenseBenchmarkIndex(
        embeddings_path=(
            "datasets/processed/"
            "fiqa_all-MiniLM-L6-v2_embeddings.npy"
        ),
        metadata_path=(
            "datasets/processed/"
            "fiqa_all-MiniLM-L6-v2_embedding_metadata.pkl"
        )
    )

    dense_index.load()

    bm25s_index = BM25SBenchmarkIndex(
        dataset_name="fiqa"
    )

    bm25s_index.load()

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    (
        baselines,
        d2rag
    ) = build_systems(
        documents_by_id=documents_by_id,
        dense_index=dense_index,
        bm25s_index=bm25s_index,
        embedding_model=embedding_model
    )

    query_ids = list(
        queries.keys()
    )[:50]

    print(
        "\nRunning fixed baselines..."
    )

    runner = BEIRBenchmarkRunner(
        queries=queries,
        qrels=qrels
    )

    baseline_results = []

    for name, retriever in (
        baselines.items()
    ):

        print(
            f"Evaluating {name}..."
        )

        results = runner.evaluate_retriever(
            name=name,
            retriever=retriever,
            query_ids=query_ids
        )

        baseline_results.extend(
            results
        )

    print(
        "\nRunning D²RAG..."
    )

    d2rag_results = runner.evaluate_d2rag(
        engine=d2rag,
        query_ids=query_ids
    )

    all_results = (
        baseline_results
        +
        d2rag_results
    )

    summary = runner.summarize(
        all_results
    )

    print("\n" + "=" * 60)
    print("FIQA 50-QUERY RESULTS")
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
            "MRR@5:",
            round(
                values["mrr_at_5"],
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

    output_path = runner.save(
        all_results,
        filename="fiqa_50_query_results.json"
    )

    print("\nResults saved to:")
    print(output_path)

    print("\n" + "=" * 60)
    print("FIQA 50-QUERY BENCHMARK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()