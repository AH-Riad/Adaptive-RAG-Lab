from dataclasses import dataclass


@dataclass
class FusionScore:
    chunk_id: str
    dense_score: float
    bm25_score: float
    hybrid_score: float


class WeightedScoreFusion:
    """
    Combines normalized dense and BM25 scores.
    """

    def __init__(self, alpha: float = 0.7):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0.0 and 1.0"
            )

        self.alpha = alpha

    def combine(
        self,
        dense_score: float,
        bm25_score: float
    ) -> float:

        return (
            self.alpha * dense_score
            +
            (1.0 - self.alpha) * bm25_score
        )

    def fuse(
        self,
        dense_scores: dict[str, float],
        bm25_scores: dict[str, float]
    ) -> list[FusionScore]:

        chunk_ids = (
            set(dense_scores.keys())
            |
            set(bm25_scores.keys())
        )

        results = []

        for chunk_id in chunk_ids:

            dense_score = dense_scores.get(
                chunk_id,
                0.0
            )

            bm25_score = bm25_scores.get(
                chunk_id,
                0.0
            )

            hybrid_score = self.combine(
                dense_score,
                bm25_score
            )

            results.append(
                FusionScore(
                    chunk_id=chunk_id,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    hybrid_score=hybrid_score
                )
            )

        results.sort(
            key=lambda result: result.hybrid_score,
            reverse=True
        )

        return results