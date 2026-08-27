from dataclasses import dataclass, field


@dataclass
class CalibratedStrategy:

    strategy: str

    combined_score: float

    recall_at_5: float

    mrr_at_5: float

    ndcg_at_5: float


@dataclass
class CalibratedPolicy:

    dataset: str

    split: str

    policy_version: str

    strategies: dict[
        str,
        CalibratedStrategy
    ] = field(
        default_factory=dict
    )

    def get_strategy(
        self,
        query_type: str
    ):

        result = self.strategies.get(
            query_type
        )

        if result is None:
            return None

        return result.strategy