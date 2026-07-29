from dataclasses import dataclass
from typing import List

from src.core import Chunk


@dataclass
class EmbeddingResult:
    """
    Represents the embedding generated for a chunk.
    """

    chunk: Chunk
    embedding: List[float]
    model_name: str