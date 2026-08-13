from enum import Enum


class RetrievalStrategy(Enum):

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class QueryType(Enum):

    FACTUAL = "factual"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    COMPARISON = "comparison"
    TECHNICAL = "technical"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class RetrievalDifficulty(Enum):

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PlannerConfidence(Enum):

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"