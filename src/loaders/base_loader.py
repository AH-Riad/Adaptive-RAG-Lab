from abc import ABC, abstractmethod
from typing import List

from src.core import Document


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """

    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """
        Load documents from the given path.
        """
        pass