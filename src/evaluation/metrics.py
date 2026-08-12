import math


class RetrievalMetrics:
    """
    Calculates standard information-retrieval metrics.
    """

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int
    ) -> float:

        retrieved = retrieved_ids[:k]

        if not retrieved:
            return 0.0

        relevant = set(relevant_ids)

        hits = sum(
            1
            for chunk_id in retrieved
            if chunk_id in relevant
        )

        return hits / len(retrieved)

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int
    ) -> float:

        if not relevant_ids:
            return 0.0

        retrieved = retrieved_ids[:k]
        relevant = set(relevant_ids)

        hits = sum(
            1
            for chunk_id in retrieved
            if chunk_id in relevant
        )

        return hits / len(relevant)

    @staticmethod
    def reciprocal_rank(
        retrieved_ids: list[str],
        relevant_ids: list[str]
    ) -> float:

        relevant = set(relevant_ids)

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1
        ):

            if chunk_id in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def dcg_at_k(
        retrieved_ids: list[str],
        relevance_scores: dict[str, int],
        k: int
    ) -> float:

        retrieved = retrieved_ids[:k]

        score = 0.0

        for rank, chunk_id in enumerate(
            retrieved,
            start=1
        ):

            relevance = relevance_scores.get(
                chunk_id,
                0
            )

            score += (
                (2 ** relevance - 1)
                / math.log2(rank + 1)
            )

        return score

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevance_scores: dict[str, int],
        k: int
    ) -> float:

        if not relevance_scores:
            return 0.0

        actual_dcg = RetrievalMetrics.dcg_at_k(
            retrieved_ids,
            relevance_scores,
            k
        )

        ideal_ids = sorted(
            relevance_scores,
            key=relevance_scores.get,
            reverse=True
        )

        ideal_dcg = RetrievalMetrics.dcg_at_k(
            ideal_ids,
            relevance_scores,
            k
        )

        if ideal_dcg == 0:
            return 0.0

        return actual_dcg / ideal_dcg