import json
from pathlib import Path


class CalibratedFeedbackPolicy:

    def __init__(
        self,
        policy_path: str,
        minimum_samples: int = 5
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

            self.data = json.load(file)

        self.policy_version = (
            self.data.get(
                "version",
                "unknown"
            )
        )

        self.minimum_samples = (
            minimum_samples
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

    def _state_key(
        self,
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

    def _fallback_state_key(
        self,
        query_type,
        strategy
    ):

        return str(
            (
                query_type,
                strategy
            )
        )

    @staticmethod
    def _is_reliable(
        result,
        minimum_samples
    ):

        if result is None:
            return False

        samples = result.get(
            "samples",
            0
        )

        return (
            samples >= minimum_samples
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

        if self._is_reliable(
            result,
            self.minimum_samples
        ):

            return result

        fallback_key = (
            self._fallback_state_key(
                query_type,
                current_strategy
            )
        )

        fallback = self.strategy_policy.get(
            fallback_key
        )

        if self._is_reliable(
            fallback,
            self.minimum_samples
        ):

            return fallback

        return None

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

        if self._is_reliable(
            result,
            self.minimum_samples
        ):

            return result

        fallback_key = (
            self._fallback_state_key(
                query_type,
                current_strategy
            )
        )

        fallback = self.topk_policy.get(
            fallback_key
        )

        if self._is_reliable(
            fallback,
            self.minimum_samples
        ):

            return fallback

        return None

    def get_state_info(
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

        return {
            "state":
                key,

            "strategy_policy":
                self.strategy_policy.get(
                    key
                ),

            "topk_policy":
                self.topk_policy.get(
                    key
                )
        }