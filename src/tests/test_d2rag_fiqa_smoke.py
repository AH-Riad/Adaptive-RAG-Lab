from src.core.adaptive_context import (
    AdaptiveContext
)

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

from src.evaluation.metrics import (
    RetrievalMetrics
)


def main():

    print("=" * 60)
    print("D²RAG FIQA SMOKE TEST")
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
        "Queries:",
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

    bm25 = BenchmarkBM25SRetriever(
        index=bm25s_index,
        documents_by_id=documents_by_id,
        top_k=5
    )

    hybrid = BenchmarkHybridRetriever(
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

    engine = D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=orchestrator
    )

    query_items = list(
        queries.items()
    )[:50]

    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    accepted = 0
    total_attempts = 0
    strategy_changes = 0

    for number, (
        query_id,
        query
    ) in enumerate(
        query_items,
        start=1
    ):

        context = AdaptiveContext(
            query=query
        )

        context = engine.run(
            context
        )

        retrieved_ids = [
            chunk.chunk_id
            for chunk
            in context.retrieval_result.retrieved_chunks
        ]

        relevant_ids = list(
            qrels.get(
                query_id,
                {}
            ).keys()
        )

        relevance_scores = qrels.get(
            query_id,
            {}
        )

        recall = (
            RetrievalMetrics.recall_at_k(
                retrieved_ids,
                relevant_ids,
                5
            )
        )

        mrr = (
            RetrievalMetrics.reciprocal_rank_at_k(
                retrieved_ids,
                relevant_ids,
                5
            )
        )

        ndcg = (
            RetrievalMetrics.ndcg_at_k(
                retrieved_ids,
                relevance_scores,
                5
            )
        )

        report = context.decision_report

        changes = len(
            report.get(
                "strategy_transitions",
                []
            )
        )

        total_recall += recall
        total_mrr += mrr
        total_ndcg += ndcg

        total_attempts += (
            report[
                "retrieval_attempts"
            ]
        )

        strategy_changes += changes

        if report[
            "adaptive_retrieval_status"
        ] == "accepted":

            accepted += 1

        print(
            f"\n[{number}/50] "
            f"Query ID: {query_id}"
        )

        print(
            "Type:",
            context.query_analysis[
                "query_type"
            ]
        )

        print(
            "Initial:",
            report[
                "initial_strategy"
            ]
        )

        print(
            "Final:",
            report[
                "final_strategy"
            ]
        )

        print(
            "Recall@5:",
            round(recall, 4)
        )

        print(
            "MRR:",
            round(mrr, 4)
        )

        print(
            "nDCG@5:",
            round(ndcg, 4)
        )

        print(
            "Attempts:",
            report[
                "retrieval_attempts"
            ]
        )

    count = len(
        query_items
    )

    print("\n" + "=" * 60)
    print("FIQA SMOKE SUMMARY")
    print("=" * 60)

    print(
        "Queries:",
        count
    )

    print(
        "Average Recall@5:",
        round(
            total_recall / count,
            4
        )
    )

    print(
        "Average MRR:",
        round(
            total_mrr / count,
            4
        )
    )

    print(
        "Average nDCG@5:",
        round(
            total_ndcg / count,
            4
        )
    )

    print(
        "Evidence Accepted:",
        accepted,
        "/",
        count
    )

    print(
        "Average Attempts:",
        round(
            total_attempts / count,
            4
        )
    )

    print(
        "Strategy Changes:",
        strategy_changes
    )

    print("\n" + "=" * 60)
    print(
        "D²RAG FIQA SMOKE TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()