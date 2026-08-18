from src.core import RetrievedChunk
from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult


class BenchmarkBM25SRetriever(BaseRetriever):

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
        query: str
    ) -> RetrievalResult:

        results = self.index.search(
            query=query,
            top_k=self.top_k
        )

        retrieved_chunks = []

        max_score = max(
            (
                result["score"]
                for result in results
            ),
            default=0.0
        )

        for result in results:

            document_id = str(
                result["document_id"]
            )

            raw_score = float(
                result["score"]
            )

            normalized_score = (
                raw_score / max_score
                if max_score > 0
                else 0.0
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
                    score=normalized_score,
                    metadata={
                        **document.metadata,
                        "raw_bm25s_score":
                            raw_score,
                        "score_type":
                            "normalized_bm25s_relevance"
                    }
                )
            )

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks
        )