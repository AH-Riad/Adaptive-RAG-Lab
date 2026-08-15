from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_dataset import BenchmarkDataset


def main():

    print("=" * 60)
    print("BENCHMARK DATASET TEST")
    print("=" * 60)

    loader = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = loader.load(
        split="test"
    )

    dataset = BenchmarkDataset(
        name="fiqa",
        corpus=corpus,
        queries=queries,
        qrels=qrels
    )

    print(
        "Dataset:",
        dataset.name
    )

    print(
        "Corpus size:",
        dataset.get_corpus_size()
    )

    print(
        "Query count:",
        dataset.get_query_count()
    )

    benchmark_queries = (
        dataset.get_queries()
    )

    first = benchmark_queries[0]

    print(
        "\nExample Query ID:",
        first.query_id
    )

    print(
        "Example Query:",
        first.query
    )

    print(
        "Relevant Documents:",
        first.relevant_documents
    )

    assert dataset.get_corpus_size() == len(
        corpus
    )

    assert dataset.get_query_count() == len(
        queries
    )

    assert len(
        benchmark_queries
    ) == len(
        queries
    )

    assert first.query_id in qrels

    print("\n" + "=" * 60)
    print("BENCHMARK DATASET TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()