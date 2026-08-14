from src.core.adaptive_context import AdaptiveContext

from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker

from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from src.vectorstore.chroma_store import ChromaVectorStore

from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.adaptive_retriever import AdaptiveRetriever

from src.analyzer.query_analyzer import QueryAnalyzer

from src.adaptation.adaptive_retrieval_orchestrator import (
    AdaptiveRetrievalOrchestrator
)

from src.adaptation.d2rag_engine import D2RAGEngine

from src.evaluation.experiment_record import (
    ExperimentRecord,
    RetrievalAttemptRecord,
    StrategyTransition
)

from src.evaluation.experiment_tracker import (
    ExperimentTracker
)


def build_system():

    loader = TextLoader()

    documents = loader.load(
        "datasets/sample_retrieval_corpus.txt"
    )

    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(
        documents
    )

    print(
        f"Loaded Documents: {len(documents)}"
    )

    print(
        f"Created Chunks: {len(chunks)}"
    )

    print(
        "\nInitializing embedding model..."
    )

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    print(
        "Creating embeddings..."
    )

    embeddings = embedding_model.encode(
        chunks
    )

    print(
        "Initializing vector store..."
    )

    vector_store = ChromaVectorStore()

    vector_store.reset()

    vector_store.add(
        embeddings
    )

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks,
        top_k=5
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        top_k=5,
        alpha=0.7
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever
    )

    orchestrator = AdaptiveRetrievalOrchestrator(
        adaptive_retriever=adaptive_retriever,
        max_retries=2
    )

    engine = D2RAGEngine(
        query_analyzer=QueryAnalyzer(),
        adaptive_retrieval_orchestrator=orchestrator
    )

    return engine


def create_experiment_record(
    experiment_number,
    query,
    context
):

    report = context.decision_report

    attempt_history = [
        RetrievalAttemptRecord(
            attempt_number=attempt[
                "attempt_number"
            ],
            strategy=attempt[
                "strategy"
            ],
            top_k=attempt[
                "top_k"
            ],
            evidence_confidence=attempt[
                "evidence_confidence"
            ],
            evidence_accepted=attempt[
                "evidence_accepted"
            ]
        )
        for attempt in report[
            "attempt_history"
        ]
    ]

    strategy_transitions = [
        StrategyTransition(
            attempt_number=transition[
                "attempt_number"
            ],
            old_strategy=transition[
                "old_strategy"
            ],
            new_strategy=transition[
                "new_strategy"
            ],
            reason=transition[
                "reason"
            ]
        )
        for transition in report[
            "strategy_transitions"
        ]
    ]

    adaptation_actions = []

    for transition in strategy_transitions:

        adaptation_actions.append(
            (
                f"{transition.old_strategy}"
                f"->{transition.new_strategy}"
            )
        )

    return ExperimentRecord(
        experiment_id=(
            f"D2RAG_{experiment_number:04d}"
        ),
        query=query,
        query_type=context.query_analysis[
            "query_type"
        ],
        initial_strategy=report[
            "initial_strategy"
        ],
        initial_top_k=report[
            "initial_top_k"
        ],
        planner_confidence=report[
            "initial_planner_confidence"
        ],
        final_strategy=report[
            "final_strategy"
        ],
        final_top_k=report[
            "final_top_k"
        ],
        evidence_confidence=report[
            "final_evidence_confidence"
        ],
        evidence_accepted=(
            context.evidence_result.accepted
        ),
        attempts=report[
            "retrieval_attempts"
        ],
        adaptive_status=report[
            "adaptive_retrieval_status"
        ],
        attempt_history=attempt_history,
        strategy_transitions=(
            strategy_transitions
        ),
        adaptation_actions=(
            adaptation_actions
        ),
        metadata={
            "embedding_model":
                "all-MiniLM-L6-v2",

            "chunk_size": 150,

            "chunk_overlap": 30,

            "hybrid_alpha": 0.7
        }
    )


def main():

    print("=" * 60)
    print("D²RAG EXPERIMENT RUN")
    print("=" * 60)

    engine = build_system()

    tracker = ExperimentTracker(
        output_path=(
            "results/logs/"
            "experiments.jsonl"
        )
    )

    queries = [
        "self-attention",

        "How does the Transformer "
        "process information?",

        "How are Transformers different "
        "from recurrent neural networks?",

        "query key value representations",

        "attention"
    ]

    for number, query in enumerate(
        queries,
        start=1
    ):

        context = AdaptiveContext(
            query=query
        )

        context = engine.run(
            context
        )

        record = create_experiment_record(
            experiment_number=number,
            query=query,
            context=context
        )

        tracker.record(
            record
        )

        print("\n" + "=" * 60)
        print(
            f"Experiment: "
            f"{record.experiment_id}"
        )
        print("=" * 60)

        print(
            "Query:",
            record.query
        )

        print(
            "Initial Strategy:",
            record.initial_strategy
        )

        print(
            "Final Strategy:",
            record.final_strategy
        )

        print(
            "Attempts:",
            record.attempts
        )

        print(
            "Status:",
            record.adaptive_status
        )

        print(
            "Evidence Confidence:",
            round(
                record.evidence_confidence,
                4
            )
        )

        print(
            "Strategy Transitions:",
            len(
                record.strategy_transitions
            )
        )

    print("\n" + "=" * 60)

    print(
        "Total Stored Experiments:",
        tracker.count()
    )

    print(
        "Experiment log:",
        tracker.output_path
    )

    print("=" * 60)


if __name__ == "__main__":
    main()