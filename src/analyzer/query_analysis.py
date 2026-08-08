from dataclasses import dataclass
from dataclasses import asdict


@dataclass
class QueryAnalysis:
    query: str
    token_count: int
    character_count: int
    query_type: str
    contains_numbers: bool
    contains_comparison: bool
    contains_question_word: bool
    suggested_top_k: int
    
    # NEW FIELDS: Confidence-Aware Planning
    query_type_confidence: float = 1.0
    complexity: str = "medium"
    complexity_confidence: float = 1.0

    def to_dict(self) -> dict:
        """Converts to dictionary so policies can use context.query_analysis.get()"""
        return asdict(self)