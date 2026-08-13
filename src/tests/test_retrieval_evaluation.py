from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)
from src.vectorstore.chroma_store import ChromaVectorStore

from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever

from src.evaluation.evaluation_dataset import (
    build_ground_truth
)
from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


def build_retrievers():

    loader = TextLoader()

    documents = loader.load(
        "datasets/sample_retrieval_corpus.txt"
    )

    print(
        f"Loaded Documents: {len(documents)}"
    )

    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(documents)

    print(
        f"Created Chunks: {len(chunks)}"
    )

    print("\nInitializing embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    print("Creating embeddings...")

    embeddings = embedding_model.encode(chunks)

    print("Initializing vector store...")

    vector_store = ChromaVectorStore()

    vector_store.reset()

    vector_store.add(embeddings)

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

    return {
        "Dense": dense_retriever,
        "BM25": bm25_retriever,
        "Hybrid": hybrid_retriever
    }


def print_results(
    retriever_name,
    results
):

    print("\n" + "=" * 60)

    print(
        f"{retriever_name} EVALUATION"
    )

    print("=" * 60)

    for result in results:

        print("\nQuery Type:", result.query_type)

        print("Query:", result.query)

        print(
            "Precision@5:",
            round(result.precision_at_5, 4)
        )

        print(
            "Recall@5:",
            round(result.recall_at_5, 4)
        )

        print(
            "MRR:",
            round(result.mrr, 4)
        )

        print(
            "nDCG@5:",
            round(result.ndcg_at_5, 4)
        )


def print_aggregate_results(
    retriever_name,
    results
):

    if not results:
        return

    precision = sum(
        result.precision_at_5
        for result in results
    ) / len(results)

    recall = sum(
        result.recall_at_5
        for result in results
    ) / len(results)

    mrr = sum(
        result.mrr
        for result in results
    ) / len(results)

    ndcg = sum(
        result.ndcg_at_5
        for result in results
    ) / len(results)

    print("\n" + "-" * 60)

    print(
        f"{retriever_name} AVERAGE PERFORMANCE"
    )

    print("-" * 60)

    print(
        "Average Precision@5:",
        round(precision, 4)
    )

    print(
        "Average Recall@5:",
        round(recall, 4)
    )

    print(
        "Average MRR:",
        round(mrr, 4)
    )

    print(
        "Average nDCG@5:",
        round(ndcg, 4)
    )


def main():

    print("=" * 60)
    print("MULTI-RETRIEVER EVALUATION")
    print("=" * 60)

    retrievers = build_retrievers()

    ground_truth = build_ground_truth()

    evaluator = RetrievalEvaluator()

    all_results = {}

    for name, retriever in retrievers.items():

        print(
            f"\nEvaluating {name}..."
        )

        results = evaluator.evaluate(
            retriever,
            ground_truth
        )

        all_results[name] = results

        print_results(
            name,
            results
        )

    print("\n")
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    for name, results in all_results.items():

        print_aggregate_results(
            name,
            results
        )

    print("\n" + "=" * 60)
    print("MULTI-RETRIEVER EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()