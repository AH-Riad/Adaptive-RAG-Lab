from __future__ import annotations


class ScoreNormalizer:
    """
    Converts raw retrieval values into a normalized
    relevance score in the range [0, 1].

    For distance-based retrieval:
        lower distance  -> higher relevance

    For similarity-based retrieval:
        higher similarity -> higher relevance
    """

    @staticmethod
    def distance_to_relevance(
        distance: float,
        scale: float = 2.0,
    ) -> float:
        """
        Convert a non-negative distance into [0, 1].

        Formula:

            relevance = 1 / (1 + distance)

        The scale parameter allows us to control how
        aggressively distance decreases relevance.
        """

        if distance < 0:
            raise ValueError(
                "Distance cannot be negative."
            )

        relevance = 1.0 / (
            1.0 + (distance / scale)
        )

        return max(
            0.0,
            min(1.0, relevance)
        )

    @staticmethod
    def similarity_to_relevance(
        similarity: float,
    ) -> float:
        """
        Normalize similarity values that are already
        expected to lie in [-1, 1].
        """

        relevance = (
            similarity + 1.0
        ) / 2.0

        return max(
            0.0,
            min(1.0, relevance)
        )