from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.dense_benchmark_index import DenseBenchmarkIndex
from src.evaluation.bm25s_benchmark_index import BM25SBenchmarkIndex
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.retrievers.benchmark_dense_retriever import BenchmarkDenseRetriever
from src.retrievers.benchmark_bm25s_retriever import BenchmarkBM25SRetriever
from src.retrievers.benchmark_hybrid_retriever import BenchmarkHybridRetriever
from src.analyzer.query_analyzer import QueryAnalyzer
from src.evaluation.action_policy_builder_v7 import DiagnosisFirstActionPolicyBuilder


def main():
    print("FIQA DEVELOPMENT ACTION POLICY V7")
    print("Policy type: diagnosis-first hierarchical failure-conditioned policy")

    dataset = BEIRDataset(name="fiqa")
    corpus, queries, qrels = dataset.load(split="dev")

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus,
    )
    documents = benchmark_corpus.to_documents()
    documents_by_id = {document.id: document for document in documents}

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

    builder = DiagnosisFirstActionPolicyBuilder(
        output_path=(
            "results/logs/"
            "fiqa_dev_action_policy_v7.json"
        ),
        cost_weight=0.10,
    )

    artifact = builder.build(
        queries=queries,
        qrels=qrels,
        query_types=query_types,
        retrievers=retrievers,
    )

    summary = artifact["training_summary"]
    print()
    print("Rejected states used:", summary["rejected_states_used"])
    print("Accepted states skipped:", summary["accepted_states_skipped"])
    print("Exact strategy states:", summary["policy_states_strategy_exact"])
    print("Exact Top-K states:", summary["policy_states_topk_exact"])
    print("Strategy query-type backoff states:", summary["strategy_backoff_states_query_type"])
    print("Strategy backoff states:", summary["strategy_backoff_states_strategy"])
    print("Strategy diagnosis backoff states:", summary["strategy_backoff_states_diagnosis"])
    print("Supported Top-K:", artifact["supported_top_k"])
    print("Saved: results/logs/fiqa_dev_action_policy_v7.json")
    print("FIQA ACTION POLICY V7 COMPLETED")


if __name__ == "__main__":
    main()
