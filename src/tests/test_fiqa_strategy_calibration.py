from src.evaluation.beir_loader import (
    BEIRDataset
)

from src.evaluation.benchmark_corpus import (
    BenchmarkCorpus
)

from src.evaluation.dense_benchmark_index import (
    DenseBenchmarkIndex
)

from src.evaluation.bm25s_benchmark_index import (
    BM25SBenchmarkIndex
)

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

from src.analyzer.query_analyzer import (
    QueryAnalyzer
)

from src.evaluation.strategy_calibrator import (
    StrategyCalibrator
)


def main():

    print("=" * 60)
    print("FIQA STRATEGY CALIBRATION")
    print("=" * 60)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = (
        dataset.load(
            split="dev"
        )
    )

    print(
        "Development queries:",
        len(queries)
    )

    benchmark_corpus = (
        BenchmarkCorpus(
            dataset_name="fiqa",
            corpus=corpus
        )
    )

    documents = (
        benchmark_corpus.to_documents()
    )

    documents_by_id = {
        document.id: document
        for document in documents
    }

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

    retrievers = {
        "dense": dense,
        "bm25": bm25s,
        "hybrid": hybrid
    }

    calibrator = StrategyCalibrator(
        query_analyzer=QueryAnalyzer(),
        retrievers=retrievers
    )

    calibration = calibrator.calibrate(
        queries=queries,
        qrels=qrels
    )

    print("\n" + "=" * 60)
    print("CALIBRATED STRATEGY POLICY")
    print("=" * 60)

    for query_type, strategy in (
        calibration["policy"].items()
    ):

        print(
            f"{query_type} -> {strategy}"
        )

    print("\n" + "=" * 60)
    print("CALIBRATION DETAILS")
    print("=" * 60)

    for query_type, details in (
        calibration["report"].items()
    ):

        print(
            f"\nQuery Type: {query_type}"
        )

        print(
            "Selected:",
            details["selected_strategy"]
        )

        for strategy, scores in (
            details["strategies"].items()
        ):

            print(
                f"  {strategy}: "
                f"Recall={scores['recall_at_5']:.4f}, "
                f"MRR={scores['mrr_at_5']:.4f}, "
                f"nDCG={scores['ndcg_at_5']:.4f}, "
                f"Combined={scores['combined_score']:.4f}"
            )

    print("\n" + "=" * 60)
    print(
        "FIQA STRATEGY CALIBRATION COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()