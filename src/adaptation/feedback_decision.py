from dataclasses import dataclass, field


@dataclass
class FeedbackDecision:
    """
    Describes how the retrieval process should adapt
    after evidence assessment.
    """

    should_retry: bool

    new_top_k: int

    change_strategy: bool

    new_strategy: str | None

    rewrite_query: bool

    rerank: bool

    reason: str

    confidence: float

    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "should_retry": self.should_retry,
            "new_top_k": self.new_top_k,
            "change_strategy": self.change_strategy,
            "new_strategy": self.new_strategy,
            "rewrite_query": self.rewrite_query,
            "rerank": self.rerank,
            "reason": self.reason,
            "confidence": self.confidence,
            "actions": self.actions,
        }