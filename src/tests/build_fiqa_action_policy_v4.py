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

from src.evaluation.action_policy_builder import (
    ActionPolicyBuilder
)


def main():

    print("=" * 60)
    print("FIQA DEVELOPMENT ACTION CALIBRATION V4")
    print("=" * 60)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = (
        dataset.load(
            split="dev"
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

    bm25_index = BM25SBenchmarkIndex(
        dataset_name="fiqa"
    )

    bm25_index.load()

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    dense = BenchmarkDenseRetriever(
        index=dense_index,
        documents_by_id=documents_by_id,
        embedding_model=embedding_model,
        top_k=5
    )

    bm25 = BenchmarkBM25SRetriever(
        index=bm25_index,
        documents_by_id=documents_by_id,
        top_k=5
    )

    hybrid = BenchmarkHybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        top_k=5,
        alpha=0.7
    )

    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid
    }

    analyzer = QueryAnalyzer()

    query_types = {}

    for query_id, query in (
        queries.items()
    ):

        query_types[
            query_id
        ] = analyzer.analyze(
            query
        )[
            "query_type"
        ]

    builder = ActionPolicyBuilder(
        output_path=(
            "results/logs/"
            "fiqa_dev_action_policy_v4.json"
        ),
        cost_weight=0.10,
        minimum_gain=0.03
    )

    artifact = builder.build(
        queries=queries,
        qrels=qrels,
        query_types=query_types,
        retrievers=retrievers
    )

    print(
        "\nStrategy policy states:",
        len(
            artifact[
                "strategy_policy"
            ]
        )
    )

    print(
        "Top-K policy states:",
        len(
            artifact[
                "topk_policy"
            ]
        )
    )

    print(
        "\nObjective:"
    )

    print(
        artifact[
            "objective"
        ]
    )

    print(
        "\nSaved:"
    )

    print(
        "results/logs/"
        "fiqa_dev_action_policy_v4.json"
    )

    print("\n" + "=" * 60)
    print(
        "FIQA ACTION POLICY V4 COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()