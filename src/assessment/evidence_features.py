import math


class EvidenceFeatureExtractor:

    FEATURE_NAMES = [
        "top1_score",
        "top3_mean",
        "top5_mean",
        "score_std",
        "top1_top2_gap",
        "top1_top5_gap",
        "retrieved_count",
        "strategy_dense",
        "strategy_bm25",
        "strategy_hybrid",
        "dense_bm25_agreement"
    ]

    def extract(
        self,
        context
    ):

        result = context.retrieval_result

        chunks = result.retrieved_chunks

        scores = [
            float(chunk.score)
            for chunk in chunks
        ]

        if not scores:

            return {
                name: 0.0
                for name in self.FEATURE_NAMES
            }

        sorted_scores = sorted(
            scores,
            reverse=True
        )

        top1 = sorted_scores[0]

        top2 = (
            sorted_scores[1]
            if len(sorted_scores) > 1
            else top1
        )

        top5_scores = sorted_scores[:5]

        top3_mean = (
            sum(
                sorted_scores[:3]
            )
            /
            min(
                3,
                len(sorted_scores)
            )
        )

        top5_mean = (
            sum(top5_scores)
            /
            len(top5_scores)
        )

        mean_score = (
            sum(scores)
            /
            len(scores)
        )

        variance = (
            sum(
                (score - mean_score) ** 2
                for score in scores
            )
            /
            len(scores)
        )

        score_std = math.sqrt(
            variance
        )

        top1_top5_gap = (
            top1 - top5_scores[-1]
        )

        plan = context.retrieval_plan

        strategy = (
            plan.strategy.value
            if plan is not None
            else ""
        )

        planner_confidence = 0.0

        if plan is not None:

            results = (
                plan.policy_results
            )

            if results:

                planner_confidence = (
                    sum(
                        item.confidence
                        for item in results.values()
                    )
                    /
                    len(results)
                )

        dense_bm25_agreement = (
            self._dense_bm25_agreement(
                chunks
            )
        )

        return {
            "top1_score": top1,

            "top3_mean": top3_mean,

            "top5_mean": top5_mean,

            "score_std": score_std,

            "top1_top2_gap": (
                top1 - top2
            ),

            "top1_top5_gap": (
                top1_top5_gap
            ),

            "retrieved_count": float(
                len(scores)
            ),

            "strategy_dense": float(
                strategy == "dense"
            ),

            "strategy_bm25": float(
                strategy == "bm25"
            ),

            "strategy_hybrid": float(
                strategy == "hybrid"
            ),

            "dense_bm25_agreement": (
                dense_bm25_agreement
            )
        }

    @staticmethod
    def _dense_bm25_agreement(
        chunks
    ):

        if not chunks:
            return 0.0

        agreements = []

        for chunk in chunks:

            metadata = chunk.metadata

            if (
                "dense_score" in metadata
                and
                "bm25_score" in metadata
            ):

                dense_score = float(
                    metadata[
                        "dense_score"
                    ]
                )

                bm25_score = float(
                    metadata[
                        "bm25_score"
                    ]
                )

                difference = abs(
                    dense_score
                    -
                    bm25_score
                )

                agreements.append(
                    1.0
                    /
                    (
                        1.0
                        +
                        difference
                    )
                )

        if not agreements:
            return 0.0

        return (
            sum(agreements)
            /
            len(agreements)
        )