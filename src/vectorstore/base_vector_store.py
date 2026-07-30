from abc import ABC, abstractmethod
from typing import List

from src.embeddings.embedding_result import EmbeddingResult
from src.core import RetrievedChunk


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, embeddings: List[EmbeddingResult]):
        pass

    @abstractmethod
    def search(self,
               query_embedding,
               top_k: int):
        pass

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def reset(self):
        pass