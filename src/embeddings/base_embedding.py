from abc import ABC, abstractmethod
from typing import List

from src.core import Chunk
from .embedding_result import EmbeddingResult


class BaseEmbedding(ABC):

    @abstractmethod
    def encode(
        self,
        chunks: List[Chunk]
    ) -> List[EmbeddingResult]:
        pass