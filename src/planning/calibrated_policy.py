import json
from pathlib import Path

from src.planning.decision_types import RetrievalStrategy


class CalibratedPolicy:

    def __init__(
        self,
        path: str
    ):

        self.path = Path(path)

        if not self.path.exists():

            raise FileNotFoundError(
                f"Calibrated policy not found: "
                f"{self.path}"
            )

        with self.path.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.data = json.load(file)

        self.dataset = self.data[
            "dataset"
        ]

        self.split = self.data[
            "split"
        ]

        self.version = self.data[
            "policy_version"
        ]

        self.policy = self.data[
            "policy"
        ]

    def get_strategy(
        self,
        query_type: str
    ) -> RetrievalStrategy:

        strategy = self.policy.get(
            query_type
        )

        if strategy is None:

            strategy = self.policy.get(
                "ambiguous",
                "dense"
            )

        return RetrievalStrategy(
            strategy
        )