from pathlib import Path
import json

from src.core.adaptive_context import AdaptiveContext
from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.dense_benchmark_index import DenseBenchmarkIndex
from src.evaluation.bm25s_benchmark_index import BM25SBenchmarkIndex
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.retrievers.benchmark_dense_retriever import BenchmarkDenseRetriever
from src.retrievers.benchmark_bm25s_retriever import BenchmarkBM25SRetriever
from src.retrievers.benchmark_hybrid_retriever import BenchmarkHybridRetriever
from src.adaptation.adaptive_retrieval_orchestrator import AdaptiveRetrievalOrchestrator
from src.adaptation.d2rag_engine import D2RAGEngine
from src.analyzer.query_analyzer import QueryAnalyzer
from src.assessment.evidence_features import EvidenceFeatureExtractor
from src.evaluation.evidence_calibrator import EvidenceCalibrator
from src.evaluation.metrics import RetrievalMetrics
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.decision_types import RetrievalStrategy


MAX_QUERIES = 50

STRATEGIES = (
    "dense",
    "bm25",
    "hybrid"
)

TOP_K_VALUES = (
    3,
    5,
    10,
    15
)

CONFIDENCE_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
)

COVERAGE_THRESHOLDS = (
    0.00,
    0.20,
    0.40,
    0.60
)

TOP1_TOP2_GAP_THRESHOLDS = (
    0.00,
    0.02,
    0.05,
    0.08
)

CALIBRATOR_PATH = (
    "results/logs/"
    "fiqa_dev_evidence_calibrator_v1.json"
)

OUTPUT_PATH = Path(
    "results/logs/"
    "fiqa_dev_evidence_gate_sweep_v1.json"
)


def minimum_evidence_count(query_type):
    if query_type in {
        "lexical",
        "technical"
    }:
        return 1

    if query_type in {
        "comparison",
        "multi_hop",
        "semantic"
    }:
        return 2

    return 1


def build_retrievers(documents_by_id):
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

    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid
    }


def build_query_type_engine():
    adaptive_orchestrator = AdaptiveRetrievalOrchestrator(
        adaptive_retriever=None,
        max_retries=0
    )

    return D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=adaptive_orchestrator
    )


def get_query_type(
    query_type_engine,
    query
):
    analysis_result = (
        query_type_engine.query_analyzer.analyze(
            query
        )
    )

    return analysis_result.get(
        "query_type",
        "ambiguous"
    )


def collect_states(
    queries,
    qrels,
    retrievers,
    query_type_engine,
    calibrator
):
    feature_extractor = EvidenceFeatureExtractor()

    states = []

    for number, (
        query_id,
        query
    ) in enumerate(
        list(queries.items())[:MAX_QUERIES],
        start=1
    ):
        query_type = get_query_type(
            query_type_engine,
            query
        )

        relevance_scores = qrels.get(
            query_id,
            {}
        )

        relevant_ids = list(
            relevance_scores.keys()
        )

        required_evidence = (
            minimum_evidence_count(
                query_type
            )
        )

        for strategy in STRATEGIES:

            retriever = retrievers[
                strategy
            ]

            original_top_k = getattr(
                retriever,
                "top_k",
                5
            )

            for top_k in TOP_K_VALUES:

                try:
                    retriever.top_k = top_k

                    retrieval_result = (
                        retriever.retrieve(
                            query
                        )
                    )

                finally:
                    retriever.top_k = (
                        original_top_k
                    )

                context = AdaptiveContext(
                    query=query
                )

                context.query_analysis = {
                    "query_type": query_type
                }

                context.retrieval_plan = (
                    RetrievalPlan(
                        strategy=(
                            RetrievalStrategy(
                                strategy
                            )
                        ),
                        top_k=top_k,
                        chunk_size=0,
                        chunk_overlap=0
                    )
                )

                context.retrieval_result = (
                    retrieval_result
                )

                features = (
                    feature_extractor.extract(
                        context
                    )
                )

                confidence = (
                    calibrator.predict_probability(
                        features
                    )
                )

                retrieved_ids = [
                    chunk.chunk_id
                    for chunk
                    in retrieval_result.retrieved_chunks
                ]

                scores = [
                    float(chunk.score)
                    for chunk
                    in retrieval_result.retrieved_chunks
                ]

                relevant_count = sum(
                    score >= 0.45
                    for score in scores
                )

                coverage = 0.0

                if scores:
                    coverage = (
                        relevant_count
                        /
                        len(scores)
                    )

                recall = (
                    RetrievalMetrics.recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        top_k
                    )
                )

                ndcg = (
                    RetrievalMetrics.ndcg_at_k(
                        retrieved_ids,
                        relevance_scores,
                        top_k
                    )
                )

                top1_top2_gap = features.get(
                    "top1_top2_gap",
                    0.0
                )

                states.append({
                    "query_id": query_id,
                    "query_type": query_type,
                    "strategy": strategy,
                    "top_k": top_k,
                    "confidence": float(
                        confidence
                    ),
                    "coverage": float(
                        coverage
                    ),
                    "relevant_count": int(
                        relevant_count
                    ),
                    "minimum_evidence": int(
                        required_evidence
                    ),
                    "top1_top2_gap": float(
                        top1_top2_gap
                    ),
                    "top1_score": float(
                        features.get(
                            "top1_score",
                            0.0
                        )
                    ),
                    "top3_mean": float(
                        features.get(
                            "top3_mean",
                            0.0
                        )
                    ),
                    "top5_mean": float(
                        features.get(
                            "top5_mean",
                            0.0
                        )
                    ),
                    "score_std": float(
                        features.get(
                            "score_std",
                            0.0
                        )
                    ),
                    "dense_bm25_agreement": float(
                        features.get(
                            "dense_bm25_agreement",
                            0.0
                        )
                    ),
                    "recall": float(
                        recall
                    ),
                    "ndcg": float(
                        ndcg
                    )
                })

        print(
            f"Collected query {number}/"
            f"{min(MAX_QUERIES, len(queries))}"
        )

    return states


def evaluate_gate(
    states,
    confidence_threshold,
    coverage_threshold,
    gap_threshold
):
    accepted_states = 0
    good_states = 0

    accepted_good = 0
    accepted_bad = 0

    rejected_good = 0
    rejected_bad = 0

    accepted_ndcg_values = []

    for state in states:

        evidence_count_passed = (
            state["relevant_count"]
            >=
            state["minimum_evidence"]
        )

        confidence_passed = (
            state["confidence"]
            >=
            confidence_threshold
        )

        coverage_passed = (
            state["coverage"]
            >=
            coverage_threshold
        )

        gap_passed = (
            state["top1_top2_gap"]
            >=
            gap_threshold
        )

        accepted = (
            confidence_passed
            and
            evidence_count_passed
            and
            coverage_passed
            and
            gap_passed
        )

        good = (
            state["ndcg"] > 0.0
        )

        if good:
            good_states += 1

        if accepted:
            accepted_states += 1

            accepted_ndcg_values.append(
                state["ndcg"]
            )

            if good:
                accepted_good += 1
            else:
                accepted_bad += 1

        else:
            if good:
                rejected_good += 1
            else:
                rejected_bad += 1

    acceptance_precision = 0.0

    if accepted_states > 0:
        acceptance_precision = (
            accepted_good
            /
            accepted_states
        )

    acceptance_recall = 0.0

    if good_states > 0:
        acceptance_recall = (
            accepted_good
            /
            good_states
        )

    false_accept_rate = 0.0

    if accepted_states > 0:
        false_accept_rate = (
            accepted_bad
            /
            accepted_states
        )

    false_reject_rate = 0.0

    if good_states > 0:
        false_reject_rate = (
            rejected_good
            /
            good_states
        )

    f1 = 0.0

    if (
        acceptance_precision
        +
        acceptance_recall
    ) > 0.0:
        f1 = (
            2.0
            *
            acceptance_precision
            *
            acceptance_recall
            /
            (
                acceptance_precision
                +
                acceptance_recall
            )
        )

    average_accepted_ndcg = 0.0

    if accepted_ndcg_values:
        average_accepted_ndcg = (
            sum(
                accepted_ndcg_values
            )
            /
            len(
                accepted_ndcg_values
            )
        )

    acceptance_rate = (
        accepted_states
        /
        len(states)
    )

    return {
        "confidence_threshold": confidence_threshold,
        "coverage_threshold": coverage_threshold,
        "gap_threshold": gap_threshold,
        "states": len(states),
        "accepted_states": accepted_states,
        "acceptance_rate": acceptance_rate,
        "good_states": good_states,
        "accepted_good": accepted_good,
        "accepted_bad": accepted_bad,
        "rejected_good": rejected_good,
        "rejected_bad": rejected_bad,
        "acceptance_precision": acceptance_precision,
        "acceptance_recall": acceptance_recall,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "f1": f1,
        "average_accepted_ndcg": average_accepted_ndcg
    }


def main():
    print(
        "D²RAG FIQA DEV EVIDENCE GATE SWEEP"
    )
    print()

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

    print(
        "Corpus:",
        len(documents)
    )

    print(
        "Dev Queries:",
        len(queries)
    )

    retrievers = build_retrievers(
        documents_by_id
    )

    query_type_engine = (
        build_query_type_engine()
    )

    calibrator = EvidenceCalibrator()

    calibrator.load(
        CALIBRATOR_PATH
    )

    states = collect_states(
        queries=queries,
        qrels=qrels,
        retrievers=retrievers,
        query_type_engine=query_type_engine,
        calibrator=calibrator
    )

    baseline = evaluate_gate(
        states=states,
        confidence_threshold=0.55,
        coverage_threshold=0.00,
        gap_threshold=0.00
    )

    results = []

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):
        for coverage_threshold in (
            COVERAGE_THRESHOLDS
        ):
            for gap_threshold in (
                TOP1_TOP2_GAP_THRESHOLDS
            ):
                result = evaluate_gate(
                    states=states,
                    confidence_threshold=(
                        confidence_threshold
                    ),
                    coverage_threshold=(
                        coverage_threshold
                    ),
                    gap_threshold=(
                        gap_threshold
                    )
                )

                results.append(
                    result
                )

    precision_candidates = sorted(
        results,
        key=lambda item: (
            item["acceptance_precision"],
            item["acceptance_recall"],
            item["average_accepted_ndcg"]
        ),
        reverse=True
    )

    f1_candidates = sorted(
        results,
        key=lambda item: (
            item["f1"],
            item["acceptance_precision"],
            item["acceptance_recall"]
        ),
        reverse=True
    )

    practical_candidates = [
        item
        for item in results
        if item["acceptance_rate"] >= 0.25
        and item["acceptance_rate"] <= 0.75
    ]

    practical_candidates = sorted(
        practical_candidates,
        key=lambda item: (
            item["f1"],
            item["acceptance_precision"],
            item["average_accepted_ndcg"]
        ),
        reverse=True
    )

    artifact = {
        "dataset": "fiqa",
        "split": "dev",
        "query_limit": MAX_QUERIES,
        "candidate_count": len(results),
        "baseline": baseline,
        "top_by_f1": f1_candidates[:10],
        "top_by_precision": precision_candidates[:10],
        "top_practical_candidates": (
            practical_candidates[:10]
        ),
        "state_count": len(states),
        "states": states
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            artifact,
            file,
            indent=2
        )

    print()
    print(
        "Baseline:"
    )
    print(
        "  Confidence:",
        baseline["confidence_threshold"]
    )
    print(
        "  Coverage:",
        baseline["coverage_threshold"]
    )
    print(
        "  Gap:",
        baseline["gap_threshold"]
    )
    print(
        "  Acceptance Precision:",
        round(
            baseline["acceptance_precision"],
            4
        )
    )
    print(
        "  Acceptance Recall:",
        round(
            baseline["acceptance_recall"],
            4
        )
    )
    print(
        "  False Accept Rate:",
        round(
            baseline["false_accept_rate"],
            4
        )
    )
    print(
        "  False Reject Rate:",
        round(
            baseline["false_reject_rate"],
            4
        )
    )
    print(
        "  F1:",
        round(
            baseline["f1"],
            4
        )
    )

    print()
    print(
        "Top practical gate candidates:"
    )

    for number, result in enumerate(
        practical_candidates[:10],
        start=1
    ):
        print(
            f"{number}. "
            f"confidence>={result['confidence_threshold']:.2f}, "
            f"coverage>={result['coverage_threshold']:.2f}, "
            f"gap>={result['gap_threshold']:.2f} | "
            f"precision={result['acceptance_precision']:.3f}, "
            f"recall={result['acceptance_recall']:.3f}, "
            f"false_accept={result['false_accept_rate']:.3f}, "
            f"false_reject={result['false_reject_rate']:.3f}, "
            f"F1={result['f1']:.3f}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()