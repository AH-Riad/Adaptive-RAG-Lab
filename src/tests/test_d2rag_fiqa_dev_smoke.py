from src.core.adaptive_context import AdaptiveContext
from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.dense_benchmark_index import DenseBenchmarkIndex
from src.evaluation.bm25s_benchmark_index import BM25SBenchmarkIndex
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.retrievers.benchmark_dense_retriever import BenchmarkDenseRetriever
from src.retrievers.benchmark_bm25s_retriever import BenchmarkBM25SRetriever
from src.retrievers.benchmark_hybrid_retriever import BenchmarkHybridRetriever
from src.retrievers.adaptive_retriever import AdaptiveRetriever
from src.analyzer.query_analyzer import QueryAnalyzer
from src.adaptation.adaptive_retrieval_orchestrator import AdaptiveRetrievalOrchestrator
from src.adaptation.d2rag_engine import D2RAGEngine
from src.assessment.evidence_features import EvidenceFeatureExtractor
from src.evaluation.metrics import RetrievalMetrics


MAX_QUERIES = 50


def build_engine(documents_by_id):
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

    embedding_model = SentenceTransformerEmbedding()

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

    return D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=orchestrator
    )


def print_false_accept_diagnostic(
    context,
    evidence,
    feature_extractor
):
    features = feature_extractor.extract(context)

    print("False-Accept Diagnostic:")
    print(
        "  Confidence:",
        round(evidence.confidence, 4)
    )
    print(
        "  Coverage:",
        round(evidence.coverage, 4)
    )
    print(
        "  Relevant Count:",
        evidence.relevant_count
    )
    print(
        "  Retrieved Count:",
        evidence.retrieved_count
    )
    print(
        "  Top1 Score:",
        round(
            features.get("top1_score", 0.0),
            4
        )
    )
    print(
        "  Top3 Mean:",
        round(
            features.get("top3_mean", 0.0),
            4
        )
    )
    print(
        "  Top5 Mean:",
        round(
            features.get("top5_mean", 0.0),
            4
        )
    )
    print(
        "  Score Std:",
        round(
            features.get("score_std", 0.0),
            4
        )
    )
    print(
        "  Top1-Top2 Gap:",
        round(
            features.get("top1_top2_gap", 0.0),
            4
        )
    )
    print(
        "  Top1-Top5 Gap:",
        round(
            features.get("top1_top5_gap", 0.0),
            4
        )
    )
    print(
        "  Dense-BM25 Agreement:",
        round(
            features.get(
                "dense_bm25_agreement",
                0.0
            ),
            4
        )
    )


def main():
    print("D²RAG FIQA DEV SMOKE TEST")
    print()

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = dataset.load(
        split="dev"
    )

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus
    )

    documents = benchmark_corpus.to_documents()

    documents_by_id = {
        document.id: document
        for document in documents
    }

    print(
        "Corpus:",
        len(documents)
    )

    print(
        "Dev Queries:",
        len(queries)
    )

    query_items = list(
        queries.items()
    )[:MAX_QUERIES]

    print(
        "Smoke Queries:",
        len(query_items)
    )

    engine = build_engine(
        documents_by_id
    )

    feature_extractor = EvidenceFeatureExtractor()

    total_recall_at_5 = 0.0
    total_mrr_at_5 = 0.0
    total_ndcg_at_5 = 0.0

    total_recall_at_final_k = 0.0
    total_ndcg_at_final_k = 0.0

    total_attempts = 0

    evidence_accepted = 0
    strategy_changes = 0
    top_k_changes = 0
    false_accepts = 0

    confidence_deltas = []

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
        evidence = context.evidence_result

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
            "final_strategy",
            initial_strategy
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

        strategy_changed = (
            initial_strategy
            != final_strategy
        )

        top_k_changed = (
            initial_top_k
            != final_top_k
        )

        strategy_changes += int(
            strategy_changed
        )

        top_k_changes += int(
            top_k_changed
        )

        total_attempts += attempts

        if report.get(
            "adaptive_retrieval_status"
        ) == "accepted":
            evidence_accepted += 1

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
            and
            final_confidence is not None
        ):
            confidence_deltas.append(
                final_confidence
                - initial_confidence
            )

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

        is_false_accept = (
            evidence.accepted
            and
            recall_at_5 == 0.0
        )

        if is_false_accept:
            false_accepts += 1

        print()
        print(
            f"[{number}/{len(query_items)}] "
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

        if confidence_deltas:
            print(
                "Confidence Delta:",
                round(
                    confidence_deltas[-1],
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

            for transition in strategy_transitions:
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

            for attempt in attempt_history:
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

        if is_false_accept:
            print_false_accept_diagnostic(
                context=context,
                evidence=evidence,
                feature_extractor=feature_extractor
            )

        print(
            "Attempts:",
            attempts
        )

    count = len(
        query_items
    )

    average_confidence_delta = 0.0

    if confidence_deltas:
        average_confidence_delta = (
            sum(confidence_deltas)
            /
            len(confidence_deltas)
        )

    adaptation_trigger_rate = (
        (
            strategy_changes
            +
            top_k_changes
        )
        /
        count
    )

    print()
    print(
        "FIQA DEV SMOKE SUMMARY"
    )

    print(
        "Queries:",
        count
    )

    print(
        "Average Recall@5:",
        round(
            total_recall_at_5
            /
            count,
            4
        )
    )

    print(
        "Average MRR@5:",
        round(
            total_mrr_at_5
            /
            count,
            4
        )
    )

    print(
        "Average nDCG@5:",
        round(
            total_ndcg_at_5
            /
            count,
            4
        )
    )

    print(
        "Average Recall@FinalK:",
        round(
            total_recall_at_final_k
            /
            count,
            4
        )
    )

    print(
        "Average nDCG@FinalK:",
        round(
            total_ndcg_at_final_k
            /
            count,
            4
        )
    )

    print(
        "Evidence Accepted:",
        evidence_accepted,
        "/",
        count
    )

    print(
        "Average Attempts:",
        round(
            total_attempts
            /
            count,
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
        "Adaptation Trigger Rate:",
        round(
            adaptation_trigger_rate,
            4
        )
    )

    print(
        "False Accepts:",
        false_accepts
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
        for action, value in sorted(
            action_counts.items()
        ):
            print(
                f"  {action}:",
                value
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
        for diagnosis, value in sorted(
            diagnosis_counts.items()
        ):
            print(
                f"  {diagnosis}:",
                value
            )
    else:
        print(
            "  none"
        )


if __name__ == "__main__":
    main()