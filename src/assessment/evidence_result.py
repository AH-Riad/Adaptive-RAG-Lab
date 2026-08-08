from dataclasses import dataclass, field


@dataclass
class EvidenceResult:
    """
    Represents the quality assessment of retrieved evidence.
    """

    accepted: bool

    confidence: float

    average_score: float

    coverage: float

    retrieved_count: int

    relevant_count: int

    threshold: float

    reasons: list[str] = field(default_factory=list)

    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "confidence": self.confidence,
            "average_score": self.average_score,
            "coverage": self.coverage,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "threshold": self.threshold,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
        }