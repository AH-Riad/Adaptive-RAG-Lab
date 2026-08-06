from dataclasses import dataclass

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

    notes: str = ""