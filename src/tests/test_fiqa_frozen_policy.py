import ast
import json
from pathlib import Path

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

from src.planning.calibrated_policy import (
    CalibratedPolicy
)

from src.assessment.evidence_features import (
    EvidenceFeatureExtractor
)

from src.evaluation.evidence_calibrator import (
    EvidenceCalibrator
)

from src.core.adaptive_context import (
    AdaptiveContext
)

from src.planning.retrieval_plan import (
    RetrievalPlan
)

from src.planning.decision_types import (
    RetrievalStrategy
)

from src.evaluation.metrics import (
    RetrievalMetrics
)

TEST_POLICY_PATH = (
    "results/logs/"
    "fiqa_dev_action_policy_v5_1.json"
)

STRATEGY_POLICY_PATH = (
    "results/logs/"
    "fiqa_dev_strategy_policy_v1.json"
)

EVIDENCE_CALIBRATOR_PATH = (
    "results/logs/"
    "fiqa_dev_evidence_calibrator_v1.json"
)

RESULT_PATH = (
    "results/logs/"
    "fiqa_test_frozen_policy_results.json"
)

TOP_K = 5

MAX_D2RAG_ACTIONS = 3


def confidence_bucket(
    confidence
):
    if confidence < 0.25:
        return "very_low"

    if confidence < 0.50:
        return "low"

    if confidence < 0.75:
        return "medium"

    return "high"


def retrieve(
    retriever,
    query,
    top_k
):
    original_top_k = getattr(
        retriever,
        "top_k",
        TOP_K
    )

    try:

        retriever.top_k = top_k

        result = retriever.retrieve(
            query
        )

    finally:

        retriever.top_k = (
            original_top_k
        )

    return result


def retrieved_ids(
    result
):
    return [
        chunk.chunk_id
        for chunk in (
            result.retrieved_chunks
        )
    ]


def evaluate_retrieval(
    retrieved,
    relevant
):
    relevant_ids = list(
        relevant.keys()
    )

    precision = (
        RetrievalMetrics.precision_at_k(
            retrieved,
            relevant_ids,
            TOP_K
        )
    )

    recall = (
        RetrievalMetrics.recall_at_k(
            retrieved,
            relevant_ids,
            TOP_K
        )
    )

    mrr = (
        RetrievalMetrics.reciprocal_rank_at_k(
            retrieved,
            relevant_ids,
            TOP_K
        )
    )

    ndcg = (
        RetrievalMetrics.ndcg_at_k(
            retrieved,
            relevant,
            TOP_K
        )
    )

    return {
        "precision_at_5": precision,
        "recall_at_5": recall,
        "mrr_at_5": mrr,
        "ndcg_at_5": ndcg
    }


def assess_evidence(
    query,
    query_type,
    strategy,
    top_k,
    retriever,
    feature_extractor,
    calibrator
):
    result = retrieve(
        retriever,
        query,
        top_k
    )

    context = AdaptiveContext(
        query=query
    )

    context.query_analysis = {
        "query_type":
            query_type
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
        result
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

    return (
        result,
        confidence,
        confidence_bucket(
            confidence
        )
    )


def parse_policy_state(
    state
):
    return ast.literal_eval(
        state
    )


def lookup_policy_action(
    policy,
    query_type,
    strategy,
    bucket,
    top_k
):
    state = (
        query_type,
        strategy,
        bucket,
        top_k
    )

    state_key = str(
        state
    )

    entry = policy.get(
        state_key
    )

    if entry is None:
        return "keep"

    return entry.get(
        "selected_action",
        "keep"
    )


def apply_strategy_action(
    action,
    current_strategy
):
    if action == "switch_to_dense":
        return "dense"

    if action == "switch_to_bm25":
        return "bm25"

    if action == "switch_to_hybrid":
        return "hybrid"

    return current_strategy


def apply_topk_action(
    action,
    current_top_k
):
    if not action.startswith(
        "set_top_k_"
    ):

        return current_top_k

    target = int(
        action.split(
            "_"
        )[-1]
    )

    if target <= current_top_k:
        return current_top_k

    return target


def mean(
    values
):
    if not values:
        return 0.0

    return (
        sum(values)
        /
        len(values)
    )


def evaluate_fixed_baseline(
    name,
    queries,
    qrels,
    retriever
):
    records = []

    for query_id, query in (
        queries.items()
    ):

        relevant = qrels.get(
            query_id,
            {}
        )

        result = retrieve(
            retriever,
            query,
            TOP_K
        )

        ids = retrieved_ids(
            result
        )

        metrics = (
            evaluate_retrieval(
                ids,
                relevant
            )
        )

        records.append({
            "query_id":
                query_id,

            "metrics":
                metrics
        })

    return records


def aggregate_records(
    records
):
    precisions = [
        row["metrics"][
            "precision_at_5"
        ]
        for row in records
    ]

    recalls = [
        row["metrics"][
            "recall_at_5"
        ]
        for row in records
    ]

    mrrs = [
        row["metrics"][
            "mrr_at_5"
        ]
        for row in records
    ]

    ndcgs = [
        row["metrics"][
            "ndcg_at_5"
        ]
        for row in records
    ]

    return {
        "queries":
            len(records),

        "precision_at_5":
            mean(precisions),

        "recall_at_5":
            mean(recalls),

        "mrr_at_5":
            mean(mrrs),

        "ndcg_at_5":
            mean(ndcgs)
    }


def evaluate_d2rag(
    queries,
    qrels,
    query_types,
    retrievers,
    strategy_policy,
    action_policy,
    feature_extractor,
    calibrator
):
    records = []

    for index, (
        query_id,
        query
    ) in enumerate(
        queries.items(),
        start=1
    ):

        query_type = query_types[
            query_id
        ]

        relevant = qrels.get(
            query_id,
            {}
        )

        current_strategy = (
            strategy_policy
            .get_strategy(
                query_type
            )
            .value
        )

        current_top_k = TOP_K

        initial_strategy = (
            current_strategy
        )

        initial_top_k = (
            current_top_k
        )

        (
            result,
            evidence_confidence,
            bucket
        ) = assess_evidence(
            query=query,
            query_type=query_type,
            strategy=current_strategy,
            top_k=current_top_k,
            retriever=retrievers[
                current_strategy
            ],
            feature_extractor=(
                feature_extractor
            ),
            calibrator=calibrator
        )

        initial_confidence = (
            evidence_confidence
        )

        strategy_changes = 0
        topk_changes = 0
        attempts = 1

        strategy_action = "keep"
        topk_action = "keep"

        trajectory = []

        final_result = result

        for _ in range(
            MAX_D2RAG_ACTIONS
        ):

            state_strategy_action = (
                lookup_policy_action(
                    action_policy[
                        "strategy_policy"
                    ],
                    query_type,
                    current_strategy,
                    bucket,
                    current_top_k
                )
            )

            if (
                state_strategy_action
                != "keep"
            ):

                new_strategy = (
                    apply_strategy_action(
                        state_strategy_action,
                        current_strategy
                    )
                )

                if new_strategy != (
                    current_strategy
                ):

                    current_strategy = (
                        new_strategy
                    )

                    strategy_changes += 1
                    attempts += 1
                    strategy_action = (
                        state_strategy_action
                    )

                    (
                        final_result,
                        evidence_confidence,
                        bucket
                    ) = assess_evidence(
                        query=query,
                        query_type=query_type,
                        strategy=current_strategy,
                        top_k=current_top_k,
                        retriever=retrievers[
                            current_strategy
                        ],
                        feature_extractor=(
                            feature_extractor
                        ),
                        calibrator=calibrator
                    )

                    trajectory.append({
                        "action":
                            state_strategy_action,

                        "strategy":
                            current_strategy,

                        "top_k":
                            current_top_k,

                        "evidence_confidence":
                            evidence_confidence,

                        "confidence_bucket":
                            bucket
                    })

                    continue

            state_topk_action = (
                lookup_policy_action(
                    action_policy[
                        "topk_policy"
                    ],
                    query_type,
                    current_strategy,
                    bucket,
                    current_top_k
                )
            )

            if (
                state_topk_action
                != "keep"
            ):

                new_top_k = (
                    apply_topk_action(
                        state_topk_action,
                        current_top_k
                    )
                )

                if new_top_k != (
                    current_top_k
                ):

                    current_top_k = (
                        new_top_k
                    )

                    topk_changes += 1
                    attempts += 1
                    topk_action = (
                        state_topk_action
                    )

                    (
                        final_result,
                        evidence_confidence,
                        bucket
                    ) = assess_evidence(
                        query=query,
                        query_type=query_type,
                        strategy=current_strategy,
                        top_k=current_top_k,
                        retriever=retrievers[
                            current_strategy
                        ],
                        feature_extractor=(
                            feature_extractor
                        ),
                        calibrator=calibrator
                    )

                    trajectory.append({
                        "action":
                            state_topk_action,

                        "strategy":
                            current_strategy,

                        "top_k":
                            current_top_k,

                        "evidence_confidence":
                            evidence_confidence,

                        "confidence_bucket":
                            bucket
                    })

                    continue

            break

        final_ids = retrieved_ids(
            final_result
        )

        metrics = (
            evaluate_retrieval(
                final_ids,
                relevant
            )
        )

        records.append({

            "query_id":
                query_id,

            "query_type":
                query_type,

            "initial_strategy":
                initial_strategy,

            "initial_top_k":
                initial_top_k,

            "initial_evidence_confidence":
                initial_confidence,

            "initial_confidence_bucket":
                confidence_bucket(
                    initial_confidence
                ),

            "strategy_action":
                strategy_action,

            "topk_action":
                topk_action,

            "final_strategy":
                current_strategy,

            "final_top_k":
                current_top_k,

            "final_evidence_confidence":
                evidence_confidence,

            "final_confidence_bucket":
                bucket,

            "attempts":
                attempts,

            "strategy_changes":
                strategy_changes,

            "topk_changes":
                topk_changes,

            "trajectory":
                trajectory,

            "metrics":
                metrics
        })

        if (
            index % 100 == 0
            or
            index == len(queries)
        ):

            print(
                f"D²RAG: "
                f"{index}/{len(queries)}"
            )

    return records


def main():
    print("=" * 70)
    print("FIQA HELD-OUT TEST: FROZEN POLICY EVALUATION")
    print("=" * 70)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = (
        dataset.load(
            split="test"
        )
    )

    print(
        f"Corpus: {len(corpus)}"
    )

    print(
        f"Queries: {len(queries)}"
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
        top_k=TOP_K
    )

    bm25 = BenchmarkBM25SRetriever(
        index=bm25_index,
        documents_by_id=documents_by_id,
        top_k=TOP_K
    )

    hybrid = BenchmarkHybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        top_k=TOP_K,
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

    strategy_policy = (
        CalibratedPolicy(
            path=STRATEGY_POLICY_PATH
        )
    )

    with open(
        TEST_POLICY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        action_policy = json.load(
            file
        )

    feature_extractor = (
        EvidenceFeatureExtractor()
    )

    evidence_calibrator = (
        EvidenceCalibrator()
    )

    evidence_calibrator.load(
        EVIDENCE_CALIBRATOR_PATH
    )

    print("\nRunning Dense baseline...")

    dense_records = (
        evaluate_fixed_baseline(
            "dense",
            queries,
            qrels,
            dense
        )
    )

    print("Running BM25S baseline...")

    bm25_records = (
        evaluate_fixed_baseline(
            "bm25",
            queries,
            qrels,
            bm25
        )
    )

    print("Running Hybrid baseline...")

    hybrid_records = (
        evaluate_fixed_baseline(
            "hybrid",
            queries,
            qrels,
            hybrid
        )
    )

    print(
        "Running Single-stage adaptive..."
    )

    adaptive_records = []

    for query_id, query in (
        queries.items()
    ):

        query_type = query_types[
            query_id
        ]

        strategy = (
            strategy_policy
            .get_strategy(
                query_type
            )
            .value
        )

        result = retrieve(
            retrievers[strategy],
            query,
            TOP_K
        )

        ids = retrieved_ids(
            result
        )

        metrics = (
            evaluate_retrieval(
                ids,
                qrels.get(
                    query_id,
                    {}
                )
            )
        )

        adaptive_records.append({
            "query_id":
                query_id,

            "strategy":
                strategy,

            "metrics":
                metrics
        })

    print(
        "Running D²RAG..."
    )

    d2rag_records = evaluate_d2rag(
        queries=queries,
        qrels=qrels,
        query_types=query_types,
        retrievers=retrievers,
        strategy_policy=strategy_policy,
        action_policy=action_policy,
        feature_extractor=feature_extractor,
        calibrator=evidence_calibrator
    )

    results = {

        "dataset":
            "fiqa",

        "split":
            "test",

        "evaluation_cutoff":
            5,

        "frozen_strategy_policy":
            STRATEGY_POLICY_PATH,

        "frozen_action_policy":
            TEST_POLICY_PATH,

        "frozen_evidence_calibrator":
            EVIDENCE_CALIBRATOR_PATH,

        "systems": {

            "dense": {
                "type":
                    "fixed_baseline",

                "metrics":
                    aggregate_records(
                        dense_records
                    )
            },

            "bm25s": {
                "type":
                    "fixed_baseline",

                "metrics":
                    aggregate_records(
                        bm25_records
                    )
            },

            "hybrid": {
                "type":
                    "fixed_baseline",

                "metrics":
                    aggregate_records(
                        hybrid_records
                    )
            },

            "single_stage_adaptive": {
                "type":
                    "frozen_strategy_policy",

                "metrics":
                    aggregate_records(
                        adaptive_records
                    )
            },

            "d2rag": {
                "type":
                    "frozen_dual_stage_policy",

                "metrics":
                    aggregate_records(
                        d2rag_records
                    )
            }
        },

        "d2rag_records":
            d2rag_records
    }

    output_path = Path(
        RESULT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print("\n" + "=" * 70)
    print("FIQA HELD-OUT TEST SUMMARY")
    print("=" * 70)

    for system_name, system in (
        results["systems"].items()
    ):

        metrics = system["metrics"]

        print(
            f"\n{system_name.upper()}"
        )

        print(
            f"Precision@5: "
            f"{metrics['precision_at_5']:.4f}"
        )

        print(
            f"Recall@5:    "
            f"{metrics['recall_at_5']:.4f}"
        )

        print(
            f"MRR@5:       "
            f"{metrics['mrr_at_5']:.4f}"
        )

        print(
            f"nDCG@5:      "
            f"{metrics['ndcg_at_5']:.4f}"
        )

    strategy_changes = sum(
        row[
            "strategy_changes"
        ]
        for row in d2rag_records
    )

    topk_changes = sum(
        row[
            "topk_changes"
        ]
        for row in d2rag_records
    )

    total_attempts = sum(
        row[
            "attempts"
        ]
        for row in d2rag_records
    )

    print("\nD²RAG behavior:")

    print(
        f"Strategy changes: "
        f"{strategy_changes}"
    )

    print(
        f"Top-K changes: "
        f"{topk_changes}"
    )

    print(
        f"Average attempts: "
        f"{total_attempts / len(d2rag_records):.3f}"
    )

    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )

    print("\n" + "=" * 70)
    print("FIQA HELD-OUT TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()