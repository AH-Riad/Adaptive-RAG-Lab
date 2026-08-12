from src.evaluation.metrics import RetrievalMetrics


def main():

    print("=" * 60)
    print("RETRIEVAL METRICS TEST")
    print("=" * 60)

    retrieved = [
        "A",
        "B",
        "C",
        "D",
        "E"
    ]

    relevant = [
        "A",
        "C",
        "F"
    ]

    relevance_scores = {
        "A": 3,
        "C": 2,
        "F": 3
    }

    precision = RetrievalMetrics.precision_at_k(
        retrieved,
        relevant,
        5
    )

    recall = RetrievalMetrics.recall_at_k(
        retrieved,
        relevant,
        5
    )

    mrr = RetrievalMetrics.reciprocal_rank(
        retrieved,
        relevant
    )

    ndcg = RetrievalMetrics.ndcg_at_k(
        retrieved,
        relevance_scores,
        5
    )

    print("\nRetrieved:")
    print(retrieved)

    print("\nRelevant:")
    print(relevant)

    print("\nPrecision@5:")
    print(round(precision, 4))

    print("\nRecall@5:")
    print(round(recall, 4))

    print("\nMRR:")
    print(round(mrr, 4))

    print("\nnDCG@5:")
    print(round(ndcg, 4))

    print("\n" + "=" * 60)
    print("METRICS TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()