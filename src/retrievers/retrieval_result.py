from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class RetrievalResult:
    """
    Stores the query and the resulting chunks from the vector database.
    """
    query: str
    retrieved_chunks: list = field(default_factory=list)