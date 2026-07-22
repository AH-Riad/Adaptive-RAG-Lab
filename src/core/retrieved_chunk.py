from dataclasses import dataclass

from .chunk import Chunk


@dataclass
class RetrievedChunk:
    """
    Represents a retrieved chunk along with retrieval metadata.
    """

    chunk: Chunk
    score: float
    rank: int
    retriever: str