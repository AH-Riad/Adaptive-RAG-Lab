from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:

    system: str
    query: str
    query_type: str

    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float

    attempts: int = 1

    initial_strategy: str = ""
    final_strategy: str = ""

    initial_top_k: int = 0
    final_top_k: int = 0

    planner_confidence: float = 0.0
    evidence_confidence: float = 0.0

    accepted: bool = False

    strategy_changes: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:

        return {
            "system": self.system,
            "query": self.query,
            "query_type": self.query_type,
            "precision_at_5": self.precision_at_5,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "ndcg_at_5": self.ndcg_at_5,
            "attempts": self.attempts,
            "initial_strategy": self.initial_strategy,
            "final_strategy": self.final_strategy,
            "initial_top_k": self.initial_top_k,
            "final_top_k": self.final_top_k,
            "planner_confidence": self.planner_confidence,
            "evidence_confidence": self.evidence_confidence,
            "accepted": self.accepted,
            "strategy_changes": self.strategy_changes
        }