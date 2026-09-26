from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.dense_benchmark_index import DenseBenchmarkIndex
from src.evaluation.bm25s_benchmark_index import BM25SBenchmarkIndex
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.retrievers.benchmark_dense_retriever import BenchmarkDenseRetriever
from src.retrievers.benchmark_bm25s_retriever import BenchmarkBM25SRetriever
from src.retrievers.benchmark_hybrid_retriever import BenchmarkHybridRetriever
from src.analyzer.query_analyzer import QueryAnalyzer
from src.evaluation.action_policy_builder import FailureConditionedActionPolicyBuilder


def main():
    print("FIQA DEVELOPMENT ACTION POLICY V6")
    print("Policy type: failure-conditioned dual-action policy")

    dataset = BEIRDataset(name="fiqa")
    corpus, queries, qrels = dataset.load(split="dev")

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus,
    )
    documents = benchmark_corpus.to_documents()
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
        ),
    )
    dense_index.load()

    bm25_index = BM25SBenchmarkIndex(dataset_name="fiqa")
    bm25_index.load()

    embedding_model = SentenceTransformerEmbedding()

    dense = BenchmarkDenseRetriever(
        index=dense_index,
        documents_by_id=documents_by_id,
        embedding_model=embedding_model,
        top_k=5,
    )

    bm25 = BenchmarkBM25SRetriever(
        index=bm25_index,
        documents_by_id=documents_by_id,
        top_k=5,
    )

    hybrid = BenchmarkHybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        top_k=5,
        alpha=0.7,
    )

    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
    }

    analyzer = QueryAnalyzer()
    query_types = {
        query_id: analyzer.analyze(query)["query_type"]
        for query_id, query in queries.items()
    }

    builder = FailureConditionedActionPolicyBuilder(
        output_path=(
            "results/logs/"
            "fiqa_dev_action_policy_v6.json"
        ),
        cost_weight=0.10,
        minimum_gain=0.03,
        minimum_query_support=5,
    )

    artifact = builder.build(
        queries=queries,
        qrels=qrels,
        query_types=query_types,
        retrievers=retrievers,
    )

    summary = artifact["training_summary"]
    print()
    print("Strategy policy states:", summary["policy_states_strategy"])
    print("Top-K policy states:", summary["policy_states_topk"])
    print("Combined policy states:", summary["policy_states_combined"])
    print("Rejected states used:", summary["rejected_states_used"])
    print("Accepted states skipped:", summary["accepted_states_skipped"])
    print("Supported Top-K:", artifact["supported_top_k"])
    print("Saved: results/logs/fiqa_dev_action_policy_v6.json")
    print("FIQA ACTION POLICY V6 COMPLETED")


if __name__ == "__main__":
    main()
