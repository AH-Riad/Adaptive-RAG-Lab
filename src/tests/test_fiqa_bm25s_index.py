from src.evaluation.beir_loader import (
    BEIRDataset
)

from src.evaluation.benchmark_corpus import (
    BenchmarkCorpus
)

from src.evaluation.bm25s_benchmark_index import (
    BM25SBenchmarkIndex
)


def main():

    print("=" * 60)
    print("FIQA BM25S INDEX TEST")
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

    print(
        "Documents:",
        len(documents)
    )

    index = BM25SBenchmarkIndex(
        dataset_name="fiqa"
    )

    print(
        "\nBuilding BM25S index..."
    )

    index.build(
        documents
    )

    print(
        "\nLoading BM25S index..."
    )

    index.load()

    query = queries["8"]

    print(
        "\nQuery:",
        query
    )

    results = index.search(
        query=query,
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

    assert len(
        index.document_ids
    ) == 57638

    assert len(results) == 5

    assert all(
        "document_id" in result
        and
        "score" in result
        for result in results
    )

    print("\n" + "=" * 60)
    print(
        "FIQA BM25S INDEX TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()