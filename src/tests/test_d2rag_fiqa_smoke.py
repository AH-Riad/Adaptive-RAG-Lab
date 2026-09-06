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

    print("=" * 70)
    print("D²RAG FIQA SMOKE TEST")
    print("=" * 70)

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

    orchestrator = AdaptiveRetrievalOrchestrator(
        adaptive_retriever=adaptive_retriever,
        max_retries=2
    )

    engine = D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=orchestrator
    )

    query_items = list(
        queries.items()
    )[:50]

    total_recall_at_5 = 0.0
    total_mrr_at_5 = 0.0
    total_ndcg_at_5 = 0.0

    total_recall_at_final_k = 0.0
    total_ndcg_at_final_k = 0.0

    accepted = 0
    total_attempts = 0

    strategy_changes = 0
    top_k_changes = 0

    successful_adaptations = 0
    confidence_improvements = []

    action_counts = {}
    diagnosis_counts = {}

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

        report = context.decision_report

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

        initial_strategy = report.get(
            "initial_strategy"
        )

        final_strategy = report.get(
            "final_strategy"
        )

        initial_top_k = report.get(
            "initial_top_k",
            5
        )

        final_top_k = report.get(
            "final_top_k",
            initial_top_k
        )

        attempts = report.get(
            "retrieval_attempts",
            1
        )

        attempt_history = report.get(
            "attempt_history",
            []
        )

        feedback_history = report.get(
            "feedback_history",
            []
        )

        strategy_transitions = report.get(
            "strategy_transitions",
            []
        )

        query_strategy_changed = (
            initial_strategy != final_strategy
        )

        query_top_k_changed = (
            initial_top_k != final_top_k
        )

        strategy_changes += int(
            query_strategy_changed
        )

        top_k_changes += int(
            query_top_k_changed
        )

        if query_top_k_changed or query_strategy_changed:
            successful_adaptations += int(
                attempts > 1
            )

        initial_confidence = None
        final_confidence = report.get(
            "final_evidence_confidence"
        )

        if attempt_history:
            initial_confidence = (
                attempt_history[0].get(
                    "evidence_confidence"
                )
            )

        if (
            initial_confidence is not None
            and final_confidence is not None
        ):
            confidence_delta = (
                final_confidence
                - initial_confidence
            )

            confidence_improvements.append(
                confidence_delta
            )
        else:
            confidence_delta = None

        for feedback in feedback_history:

            action = feedback.get(
                "action"
            )

            diagnosis = feedback.get(
                "diagnosis"
            )

            if action:
                action_counts[action] = (
                    action_counts.get(
                        action,
                        0
                    )
                    + 1
                )

            if diagnosis:
                diagnosis_counts[diagnosis] = (
                    diagnosis_counts.get(
                        diagnosis,
                        0
                    )
                    + 1
                )

        recall_at_5 = (
            RetrievalMetrics.recall_at_k(
                retrieved_ids,
                relevant_ids,
                5
            )
        )

        mrr_at_5 = (
            RetrievalMetrics.reciprocal_rank_at_k(
                retrieved_ids,
                relevant_ids,
                5
            )
        )

        ndcg_at_5 = (
            RetrievalMetrics.ndcg_at_k(
                retrieved_ids,
                relevance_scores,
                5
            )
        )

        evaluation_k = max(
            1,
            min(
                final_top_k,
                len(retrieved_ids)
            )
        )

        recall_at_final_k = (
            RetrievalMetrics.recall_at_k(
                retrieved_ids,
                relevant_ids,
                evaluation_k
            )
        )

        ndcg_at_final_k = (
            RetrievalMetrics.ndcg_at_k(
                retrieved_ids,
                relevance_scores,
                evaluation_k
            )
        )

        total_recall_at_5 += recall_at_5
        total_mrr_at_5 += mrr_at_5
        total_ndcg_at_5 += ndcg_at_5

        total_recall_at_final_k += (
            recall_at_final_k
        )

        total_ndcg_at_final_k += (
            ndcg_at_final_k
        )

        total_attempts += attempts

        if report.get(
            "adaptive_retrieval_status"
        ) == "accepted":

            accepted += 1

        print()
        print(
            f"[{number}/50] "
            f"Query ID: {query_id}"
        )

        print(
            "Type:",
            context.query_analysis.get(
                "query_type",
                "unknown"
            )
        )

        print(
            "Initial Strategy:",
            initial_strategy
        )

        print(
            "Final Strategy:",
            final_strategy
        )

        print(
            "Initial Top-K:",
            initial_top_k
        )

        print(
            "Final Top-K:",
            final_top_k
        )

        print(
            "Recall@5:",
            round(
                recall_at_5,
                4
            )
        )

        print(
            "MRR@5:",
            round(
                mrr_at_5,
                4
            )
        )

        print(
            "nDCG@5:",
            round(
                ndcg_at_5,
                4
            )
        )

        print(
            f"Recall@FinalK "
            f"(K={evaluation_k}):",
            round(
                recall_at_final_k,
                4
            )
        )

        print(
            f"nDCG@FinalK "
            f"(K={evaluation_k}):",
            round(
                ndcg_at_final_k,
                4
            )
        )

        if confidence_delta is not None:
            print(
                "Confidence Delta:",
                round(
                    confidence_delta,
                    4
                )
            )

        if feedback_history:

            print(
                "Feedback Actions:"
            )

            for feedback in feedback_history:

                print(
                    "  -",
                    feedback.get(
                        "action"
                    ),
                    "| diagnosis:",
                    feedback.get(
                        "diagnosis"
                    ),
                    "| expected improvement:",
                    round(
                        feedback.get(
                            "expected_improvement",
                            0.0
                        ),
                        4
                    )
                )

        else:

            print(
                "Feedback Actions: none"
            )

        if strategy_transitions:

            print(
                "Strategy Transitions:"
            )

            for transition in (
                strategy_transitions
            ):

                print(
                    "  -",
                    transition.get(
                        "old_strategy"
                    ),
                    "->",
                    transition.get(
                        "new_strategy"
                    )
                )

        if attempt_history:

            print(
                "Attempt Trajectory:"
            )

            for attempt in (
                attempt_history
            ):

                print(
                    "  - Attempt",
                    attempt.get(
                        "attempt_number"
                    ),
                    ":",
                    attempt.get(
                        "strategy"
                    ),
                    f"K={attempt.get('top_k')}",
                    "| confidence=",
                    round(
                        attempt.get(
                            "evidence_confidence",
                            0.0
                        ),
                        4
                    ),
                    "| accepted=",
                    attempt.get(
                        "evidence_accepted"
                    )
                )

        print(
            "Attempts:",
            attempts
        )

    count = len(
        query_items
    )

    average_confidence_delta = 0.0

    if confidence_improvements:

        average_confidence_delta = (
            sum(
                confidence_improvements
            )
            / len(
                confidence_improvements
            )
        )

    adaptation_rate = (
        successful_adaptations / count
    )

    print()
    print("=" * 70)
    print("FIQA SMOKE SUMMARY")
    print("=" * 70)

    print(
        "Queries:",
        count
    )

    print(
        "Average Recall@5:",
        round(
            total_recall_at_5 / count,
            4
        )
    )

    print(
        "Average MRR@5:",
        round(
            total_mrr_at_5 / count,
            4
        )
    )

    print(
        "Average nDCG@5:",
        round(
            total_ndcg_at_5 / count,
            4
        )
    )

    print(
        "Average Recall@FinalK:",
        round(
            total_recall_at_final_k / count,
            4
        )
    )

    print(
        "Average nDCG@FinalK:",
        round(
            total_ndcg_at_final_k / count,
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

    print(
        "Top-K Changes:",
        top_k_changes
    )

    print(
        "Adaptation Rate:",
        round(
            adaptation_rate,
            4
        )
    )

    print(
        "Average Confidence Delta:",
        round(
            average_confidence_delta,
            4
        )
    )

    print()
    print(
        "Feedback Action Counts:"
    )

    if action_counts:

        for action, count_value in sorted(
            action_counts.items()
        ):

            print(
                f"  {action}:",
                count_value
            )

    else:

        print(
            "  none"
        )

    print()
    print(
        "Diagnosis Counts:"
    )

    if diagnosis_counts:

        for diagnosis, count_value in sorted(
            diagnosis_counts.items()
        ):

            print(
                f"  {diagnosis}:",
                count_value
            )

    else:

        print(
            "  none"
        )

    print()
    print("=" * 70)
    print(
        "D²RAG FIQA SMOKE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()