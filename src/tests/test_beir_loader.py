from src.evaluation.beir_loader import BEIRDataset


def main():

    print("=" * 60)
    print("BEIR LOADER SMOKE TEST")
    print("=" * 60)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = (
        dataset.load(
            split="test"
        )
    )

    print(
        "Corpus documents:",
        len(corpus)
    )

    print(
        "Queries:",
        len(queries)
    )

    print(
        "Queries with qrels:",
        len(qrels)
    )

    first_document_id = next(
        iter(corpus)
    )

    first_query_id = next(
        iter(queries)
    )

    print(
        "Example document ID:",
        first_document_id
    )

    print(
        "Example query ID:",
        first_query_id
    )

    print(
        "Example query:",
        queries[first_query_id]
    )

    print("\n" + "=" * 60)
    print("BEIR LOADER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()