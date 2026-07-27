from abc import ABC, abstractmethod
from typing import List

from src.core import Document, Chunk


class BaseChunker(ABC):
    """
    Base class for all chunking strategies.
    """

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Chunk]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects.

        Returns:
            List of Chunk objects.
        """
        pass