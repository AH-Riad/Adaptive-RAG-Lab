from src.evaluation.dense_benchmark_index import (
    DenseBenchmarkIndex
)

from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)


def main():

    print("=" * 60)
    print("FIQA DENSE BENCHMARK INDEX TEST")
    print("=" * 60)

    index = DenseBenchmarkIndex(
        embeddings_path=(
            "datasets/processed/"
            "fiqa_all-MiniLM-L6-v2_embeddings.npy"
        ),
        metadata_path=(
            "datasets/processed/"
            "fiqa_all-MiniLM-L6-v2_embedding_metadata.pkl"
        )
    )

    index.load()

    print(
        "Embedding shape:",
        index.embeddings.shape
    )

    print(
        "Document IDs:",
        len(index.document_ids)
    )

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    query = (
        "How to deposit a cheque "
        "issued to an associate in "
        "my business into my business account?"
    )

    print(
        "\nQuery:",
        query
    )

    results = index.search(
        query_embedding=(
            embedding_model.encode_query(
                query
            )
        ),
        top_k=5
    )

    print(
        "\nTop results:"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['document_id']} "
            f"-> "
            f"{result['score']:.6f}"
        )

    assert (
        index.embeddings.shape
        ==
        (57638, 384)
    )

    assert (
        len(index.document_ids)
        ==
        57638
    )

    assert len(results) == 5

    assert all(
        "document_id" in result
        and
        "score" in result
        for result in results
    )

    print("\n" + "=" * 60)
    print(
        "DENSE BENCHMARK INDEX TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()