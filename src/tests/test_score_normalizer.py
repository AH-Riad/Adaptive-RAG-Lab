from src.retrievers.score_normalizer import ScoreNormalizer


def main():

    print("=" * 60)
    print("DISTANCE → RELEVANCE NORMALIZATION")
    print("=" * 60)

    distances = [
        0.0,
        0.25,
        0.5,
        1.0,
        1.5,
        2.0,
    ]

    previous = 1.0

    for distance in distances:

        relevance = (
            ScoreNormalizer.distance_to_relevance(
                distance
            )
        )

        print(
            f"Distance: {distance:<6} "
            f"Relevance: {relevance:.4f}"
        )

        assert 0.0 <= relevance <= 1.0

        if distance > 0:
            assert relevance < previous

        previous = relevance

    print("=" * 60)
    print("ALL NORMALIZATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()