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

from src.retrievers.policy_routed_retriever import (
    PolicyRoutedRetriever
)

from src.evaluation.beir_benchmark_runner import (
    BEIRBenchmarkRunner
)


def main():

    print("=" * 60)
    print("FIQA PRE-DECISION ONLY ABLATION")
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
        "Test queries:",
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

    predecision = PolicyRoutedRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25s,
        hybrid_retriever=hybrid,
        policy_path=(
            "results/logs/"
            "fiqa_dev_strategy_policy_v1.json"
        )
    )

    runner = BEIRBenchmarkRunner(
        queries=queries,
        qrels=qrels
    )

    print(
        "\nRunning Pre-Decision Only..."
    )

    results = runner.evaluate_retriever(
        name="PreDecision",
        retriever=predecision,
        query_ids=list(
            queries.keys()
        )
    )

    summary = runner.summarize(
        results
    )

    print("\n" + "=" * 60)
    print("PRE-DECISION ONLY RESULTS")
    print("=" * 60)

    values = summary[
        "PreDecision"
    ]

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

    path = runner.save(
        results,
        filename=(
            "fiqa_predecision_test_results.json"
        )
    )

    print(
        "\nResults saved to:"
    )

    print(path)

    print("\n" + "=" * 60)
    print(
        "PRE-DECISION ABLATION COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()