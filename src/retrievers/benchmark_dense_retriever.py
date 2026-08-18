from src.core import RetrievedChunk
from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult


class BenchmarkDenseRetriever(BaseRetriever):

    def __init__(
        self,
        index,
        documents_by_id,
        top_k: int = 5
    ):
        self.index = index
        self.documents_by_id = documents_by_id
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        query_embedding
    ) -> RetrievalResult:

        results = self.index.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        retrieved_chunks = []

        for result in results:

            document_id = str(
                result["document_id"]
            )

            document = (
                self.documents_by_id[
                    document_id
                ]
            )

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=document_id,
                    text=document.text,
                    score=float(
                        result["score"]
                    ),
                    metadata={
                        **document.metadata,
                        "raw_dense_score":
                            result["score"],
                        "score_type":
                            "dense_similarity"
                    }
                )
            )

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks
        )