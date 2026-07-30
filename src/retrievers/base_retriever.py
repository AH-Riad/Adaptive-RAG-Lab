from abc import ABC, abstractmethod

from src.retrievers.retrieval_result import RetrievalResult


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str
    ) -> RetrievalResult:
        pass