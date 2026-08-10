from src.retrievers.score_fusion import WeightedScoreFusion


def main():

    print("=" * 60)
    print("SCORE FUSION TEST")
    print("=" * 60)

    dense_scores = {
        "chunk_1": 0.80,
        "chunk_2": 0.40,
        "chunk_3": 0.20
    }

    bm25_scores = {
        "chunk_1": 0.20,
        "chunk_2": 0.90,
        "chunk_3": 0.10
    }

    fusion = WeightedScoreFusion(
        alpha=0.7
    )

    results = fusion.fuse(
        dense_scores=dense_scores,
        bm25_scores=bm25_scores
    )

    print("\nAlpha:", fusion.alpha)

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {index}"
        )

        print(
            "ID:",
            result.chunk_id
        )

        print(
            "Dense:",
            round(result.dense_score, 4)
        )

        print(
            "BM25:",
            round(result.bm25_score, 4)
        )

        print(
            "Hybrid:",
            round(result.hybrid_score, 4)
        )

    print("\n" + "=" * 60)
    print("SCORE FUSION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()