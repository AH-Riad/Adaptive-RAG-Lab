from dataclasses import dataclass


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