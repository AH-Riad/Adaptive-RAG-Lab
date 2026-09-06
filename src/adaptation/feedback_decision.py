from dataclasses import dataclass, field

@dataclass
class FeedbackDecision:
    action: str
    reason: str
    confidence: float
    source: str = "heuristic"
    target_strategy: str | None = None
    target_top_k: int | None = None
    diagnosis: str = "unknown"
    expected_improvement: float = 0.0
    action_cost: float = 0.0
    should_retry: bool = False
    previous_confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
            "target_strategy": self.target_strategy,
            "target_top_k": self.target_top_k,
            "diagnosis": self.diagnosis,
            "expected_improvement": self.expected_improvement,
            "action_cost": self.action_cost,
            "should_retry": self.should_retry,
            "previous_confidence": self.previous_confidence,
            "metadata": self.metadata
        }