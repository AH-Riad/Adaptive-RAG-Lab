from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
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


def main():

    print("=" * 60)
    print("FIQA BENCHMARK RETRIEVER TEST")
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

    query = queries["8"]

    print(
        "\nQuery:",
        query
    )

    query_embedding = (
        embedding_model.encode_query(
            query
        )
    )

    dense_result = dense.retrieve(
        query=query,
        query_embedding=query_embedding
    )

    bm25_result = bm25.retrieve(
        query
    )

    hybrid_result = hybrid.retrieve(
        query=query,
        query_embedding=query_embedding
    )

    for name, result in [
        ("DENSE", dense_result),
        ("BM25S", bm25_result),
        ("HYBRID", hybrid_result)
    ]:

        print(
            "\n" + "=" * 60
        )

        print(
            name
        )

        print(
            "=" * 60
        )

        for rank, chunk in enumerate(
            result.retrieved_chunks,
            start=1
        ):

            print(
                rank,
                chunk.chunk_id,
                "->",
                round(
                    chunk.score,
                    6
                )
            )

    print("\n" + "=" * 60)
    print(
        "FIQA BENCHMARK RETRIEVER TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()