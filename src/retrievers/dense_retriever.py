from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.core.retrieved_chunk import RetrievedChunk


class DenseRetriever(BaseRetriever):

    def __init__(
        self,
        embedding_model,
        vector_store,
        top_k
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k