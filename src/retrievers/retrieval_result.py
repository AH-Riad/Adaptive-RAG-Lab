from dataclasses import dataclass, field
from time import time


@dataclass
class RetrievalResult:
    """
    Stores the complete retrieval execution.
    """

    strategy: str

    top_k: int

    retrieved_count: int

    retrieval_time: float

    average_score: float

    documents: list = field(default_factory=list)