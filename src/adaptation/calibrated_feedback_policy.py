import json
from pathlib import Path


class CalibratedFeedbackPolicy:

    def __init__(
        self,
        policy_path: str
    ):

        self.policy_path = Path(
            policy_path
        )

        if not self.policy_path.exists():

            raise FileNotFoundError(
                f"Feedback policy not found: "
                f"{self.policy_path}"
            )

        with self.policy_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.data = json.load(
                file
            )

        self.policy_version = (
            self.data.get(
                "version",
                "unknown"
            )
        )

        self.strategy_policy = (
            self.data.get(
                "strategy_policy",
                {}
            )
        )

        self.topk_policy = (
            self.data.get(
                "topk_policy",
                {}
            )
        )

    @staticmethod
    def _state_key(
        query_type,
        strategy,
        confidence_bucket,
        top_k
    ):

        return str(
            (
                query_type,
                strategy,
                confidence_bucket,
                top_k
            )
        )

    @staticmethod
    def _fallback_state_key(
        query_type,
        strategy
    ):

        return str(
            (
                query_type,
                strategy
            )
        )

    def get_strategy_action(
        self,
        query_type,
        current_strategy,
        confidence_bucket,
        top_k
    ):

        key = self._state_key(
            query_type,
            current_strategy,
            confidence_bucket,
            top_k
        )

        result = self.strategy_policy.get(
            key
        )

        if result is not None:

            return result

        fallback = self._fallback_state_key(
            query_type,
            current_strategy
        )

        result = self.strategy_policy.get(
            fallback
        )

        return result

    def get_topk_action(
        self,
        query_type,
        current_strategy,
        confidence_bucket,
        top_k
    ):

        key = self._state_key(
            query_type,
            current_strategy,
            confidence_bucket,
            top_k
        )

        result = self.topk_policy.get(
            key
        )

        if result is not None:

            return result

        fallback = self._fallback_state_key(
            query_type,
            current_strategy
        )

        return self.topk_policy.get(
            fallback
        )