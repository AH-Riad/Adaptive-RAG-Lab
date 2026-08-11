from src.evaluation.evaluation_dataset import (
    build_ground_truth
)


def main():

    print("=" * 60)
    print("GROUND TRUTH DATASET TEST")
    print("=" * 60)

    dataset = build_ground_truth()

    queries = dataset.get_all()

    print(
        f"\nTotal Queries: {len(queries)}"
    )

    for item in queries:

        print("\n" + "-" * 60)

        print(
            "Query Type:",
            item.query_type
        )

        print(
            "Query:",
            item.query
        )

        print(
            "Relevant Chunks:"
        )

        for chunk_id in item.relevant_chunks:

            print(
                f"  {chunk_id}"
            )

        print(
            "Relevance Scores:",
            item.relevance_scores
        )

    print("\n" + "=" * 60)
    print("GROUND TRUTH TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()