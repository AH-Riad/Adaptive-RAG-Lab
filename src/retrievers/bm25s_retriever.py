import bm25s

from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.core import RetrievedChunk


class BM25SRetriever(BaseRetriever):

    def __init__(
        self,
        documents,
        top_k: int = 5
    ):
        self.documents = documents
        self.top_k = top_k

        self.corpus_texts = [
            document.text
            for document in documents
        ]

        self.tokenized_corpus = (
            bm25s.tokenize(
                self.corpus_texts
            )
        )

        self.retriever = bm25s.BM25(
            method="lucene"
        )

        self.retriever.index(
            self.tokenized_corpus
        )

    def retrieve(
        self,
        query: str
    ) -> RetrievalResult:

        query_tokens = bm25s.tokenize(
            [query]
        )

        results, scores = (
            self.retriever.retrieve(
                query_tokens,
                k=self.top_k
            )
        )

        retrieved_chunks = []

        for index in range(
            results.shape[1]
        ):

            document_index = int(
                results[0, index]
            )

            raw_score = float(
                scores[0, index]
            )

            normalized_score = (
                raw_score
                /
                float(
                    scores[0].max()
                )
                if float(
                    scores[0].max()
                ) > 0
                else 0.0
            )

            document = self.documents[
                document_index
            ]

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=document.id,
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