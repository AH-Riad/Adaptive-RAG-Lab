from enum import Enum


class RetrievalStrategy(Enum):
    """
    Supported retrieval strategies.
    """

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class QueryType(Enum):
    """
    High-level query categories.
    """

    FACTUAL = "factual"
    SEMANTIC = "semantic"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class RetrievalDifficulty(Enum):
    """
    Estimated difficulty before retrieval.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PlannerConfidence(Enum):
    """
    Confidence assigned by the Decision Engine.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"