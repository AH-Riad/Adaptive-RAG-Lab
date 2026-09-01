from dataclasses import dataclass


@dataclass
class FeedbackDecision:

    action: str

    reason: str

    confidence: float

    source: str = "heuristic"

    target_strategy: str | None = None

    target_top_k: int | None = None