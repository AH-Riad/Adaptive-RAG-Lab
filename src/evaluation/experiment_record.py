from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalAttemptRecord:
    attempt_number: int
    strategy: str
    top_k: int
    evidence_confidence: float
    evidence_accepted: bool

    def to_dict(self) -> dict:

        return {
            "attempt_number": self.attempt_number,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "evidence_confidence":
                self.evidence_confidence,
            "evidence_accepted":
                self.evidence_accepted
        }


@dataclass
class StrategyTransition:
    attempt_number: int
    old_strategy: str
    new_strategy: str
    reason: str

    def to_dict(self) -> dict:

        return {
            "attempt_number":
                self.attempt_number,
            "old_strategy":
                self.old_strategy,
            "new_strategy":
                self.new_strategy,
            "reason":
                self.reason
        }


@dataclass
class ExperimentRecord:

    experiment_id: str

    query: str

    query_type: str

    initial_strategy: str

    initial_top_k: int

    planner_confidence: float

    final_strategy: str

    final_top_k: int

    evidence_confidence: float

    evidence_accepted: bool

    attempts: int

    adaptive_status: str

    attempt_history: list[
        RetrievalAttemptRecord
    ] = field(default_factory=list)

    strategy_transitions: list[
        StrategyTransition
    ] = field(default_factory=list)

    adaptation_actions: list[str] = field(
        default_factory=list
    )

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:

        return {
            "experiment_id":
                self.experiment_id,

            "query":
                self.query,

            "query_type":
                self.query_type,

            "initial_strategy":
                self.initial_strategy,

            "initial_top_k":
                self.initial_top_k,

            "planner_confidence":
                self.planner_confidence,

            "final_strategy":
                self.final_strategy,

            "final_top_k":
                self.final_top_k,

            "evidence_confidence":
                self.evidence_confidence,

            "evidence_accepted":
                self.evidence_accepted,

            "attempts":
                self.attempts,

            "adaptive_status":
                self.adaptive_status,

            "attempt_history": [
                attempt.to_dict()
                for attempt
                in self.attempt_history
            ],

            "strategy_transitions": [
                transition.to_dict()
                for transition
                in self.strategy_transitions
            ],

            "adaptation_actions":
                self.adaptation_actions,

            "metrics":
                self.metrics,

            "metadata":
                self.metadata
        }