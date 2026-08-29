import json
from collections import defaultdict
from pathlib import Path


class AdaptiveActionCalibrator:

    ACTIONS = [
        "keep",
        "increase_top_k",
        "switch_to_dense",
        "switch_to_bm25",
        "switch_to_hybrid"
    ]

    def __init__(
        self,
        output_path: str
    ):

        self.output_path = Path(
            output_path
        )

        self.policy = {}

    def fit(
        self,
        samples
    ):

        grouped = defaultdict(
            list
        )

        for sample in samples:

            state = self._state_key(
                sample
            )

            grouped[state].append(
                sample
            )

        for state, rows in (
            grouped.items()
        ):

            action_scores = defaultdict(
                list
            )

            for row in rows:

                action_scores[
                    row["action"]
                ].append(
                    row["utility"]
                )

            best_action = None
            best_score = float(
                "-inf"
            )

            for action, utilities in (
                action_scores.items()
            ):

                average_utility = (
                    sum(utilities)
                    /
                    len(utilities)
                )

                if average_utility > best_score:

                    best_score = (
                        average_utility
                    )

                    best_action = action

            self.policy[state] = {
                "action": best_action,
                "utility": best_score,
                "samples": len(rows)
            }

        return self

    @staticmethod
    def _state_key(
        sample
    ):

        return (
            sample["query_type"],
            sample["current_strategy"],
            sample["confidence_bucket"],
            sample["top_k"]
        )

    def get_action(
        self,
        query_type,
        current_strategy,
        confidence,
        top_k
    ):

        confidence_bucket = (
            self._confidence_bucket(
                confidence
            )
        )

        state = (
            query_type,
            current_strategy,
            confidence_bucket,
            top_k
        )

        result = self.policy.get(
            state
        )

        if result is None:

            return "increase_top_k"

        return result["action"]

    @staticmethod
    def _confidence_bucket(
        confidence
    ):

        if confidence < 0.25:
            return "very_low"

        if confidence < 0.50:
            return "low"

        if confidence < 0.75:
            return "medium"

        return "high"

    def save(
        self,
        dataset,
        split,
        version
    ):

        artifact = {
            "dataset":
                dataset,

            "split":
                split,

            "version":
                version,

            "policy": {
                str(key):
                    value
                for key, value
                in self.policy.items()
            }
        }

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                artifact,
                file,
                indent=2
            )