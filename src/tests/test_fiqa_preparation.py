from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import (
    BenchmarkCorpus
)


def main():

    print("=" * 60)
    print("FIQA BENCHMARK PREPARATION")
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
        "BEIR corpus:",
        len(corpus)
    )

    print(
        "Document objects:",
        len(documents)
    )

    first = documents[0]

    print(
        "\nFirst document ID:",
        first.id
    )

    print(
        "Source:",
        first.source
    )

    print(
        "Benchmark ID:",
        first.metadata[
            "benchmark_id"
        ]
    )

    print(
        "Text preview:"
    )

    print(
        first.text[:300]
    )

    assert len(documents) == len(corpus)

    assert (
        first.id
        ==
        first.metadata["benchmark_id"]
    )

    print("\n" + "=" * 60)
    print(
        "FIQA BENCHMARK PREPARATION PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()