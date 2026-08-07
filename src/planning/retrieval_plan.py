from dataclasses import dataclass, field

from src.planning.decision_types import (
    PlannerConfidence,
    RetrievalDifficulty,
    RetrievalStrategy,
)


@dataclass
class RetrievalPlan:
    """
    Output produced by the Decision Engine.
    """

    strategy: RetrievalStrategy

    top_k: int

    chunk_size: int

    chunk_overlap: int

    rerank: bool = False

    rewrite_query: bool = False

    expand_query: bool = False

    difficulty: RetrievalDifficulty = RetrievalDifficulty.MEDIUM

    planner_confidence: PlannerConfidence = PlannerConfidence.MEDIUM

    decision_trace: list[str] = field(default_factory=list)
    
    policy_confidence: dict[str, float] = field(default_factory=dict)
    policy_reasons: dict[str, str] = field(default_factory=dict)
    selected_policies: list[str] = field(default_factory=list)