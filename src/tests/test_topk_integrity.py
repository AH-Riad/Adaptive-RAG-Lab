from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.dense_benchmark_index import DenseBenchmarkIndex
from src.evaluation.bm25s_benchmark_index import BM25SBenchmarkIndex
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.retrievers.benchmark_dense_retriever import BenchmarkDenseRetriever
from src.retrievers.benchmark_bm25s_retriever import BenchmarkBM25SRetriever
from src.retrievers.benchmark_hybrid_retriever import BenchmarkHybridRetriever


TOP_K_VALUES = (3, 5, 10, 15)


def count_results(result):
    chunks = getattr(result, "retrieved_chunks", None)
    return len(chunks) if chunks is not None else 0


def test_retriever(name, retriever, query):
    print(name)

    for top_k in TOP_K_VALUES:
        retriever.top_k = top_k
        result = retriever.retrieve(query)
        actual = count_results(result)
        status = "OK" if actual == top_k else "MISMATCH"
        print(
            f"  requested={top_k} actual={actual} status={status}"
        )

    print()


def main():
    dataset = BEIRDataset(name="fiqa")
    corpus, queries, _ = dataset.load(split="dev")

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus
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

    query_id, query = next(iter(queries.items()))

    print("Top-K Integrity Test")
    print("Query ID:", query_id)
    print()

    test_retriever("Dense", dense, query)
    test_retriever("BM25", bm25, query)
    test_retriever("Hybrid", hybrid, query)


if __name__ == "__main__":
    main()