from dataclasses import dataclass
from typing import List

from src.core import RetrievedChunk


@dataclass
class RetrievalResult:
    query: str
    retrieved_chunks: List[RetrievedChunk]