from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.core import RetrievedChunk


class BM25Retriever(BaseRetriever):
    """
    Lexical retrieval using BM25.

    BM25 is particularly useful for:
    - exact terms
    - course codes
    - names
    - technical keywords
    - identifiers
    """

    def __init__(
        self,
        documents,
        top_k: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.top_k = top_k
        self.k1 = k1
        self.b = b

        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self):

        self.tokenized_documents = [
            self._tokenize(document["text"])
            for document in self.documents
        ]

        self.document_count = len(
            self.tokenized_documents
        )

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.average_document_length = (
            sum(self.document_lengths)
            / self.document_count
            if self.document_count
            else 0
        )

        self.document_frequency = {}

        for tokens in self.tokenized_documents:

            unique_terms = set(tokens)

            for term in unique_terms:

                self.document_frequency[term] = (
                    self.document_frequency.get(
                        term,
                        0
                    ) + 1
                )

    def _idf(self, term: str) -> float:

        document_frequency = (
            self.document_frequency.get(term, 0)
        )

        if document_frequency == 0:
            return 0.0

        return max(
            0.0,
            (
                (
                    self.document_count
                    - document_frequency
                    + 0.5
                )
                /
                (
                    document_frequency
                    + 0.5
                )
            )
        )

    def _score_document(
        self,
        query_terms,
        document_index: int,
    ) -> float:

        document_terms = (
            self.tokenized_documents[
                document_index
            ]
        )

        document_length = (
            self.document_lengths[
                document_index
            ]
        )

        score = 0.0

        for term in query_terms:

            frequency = document_terms.count(term)

            if frequency == 0:
                continue

            idf = self._idf(term)

            numerator = (
                frequency
                * (self.k1 + 1)
            )

            denominator = (
                frequency
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.average_document_length
                    )
                )
            )

            score += (
                idf
                * numerator
                / denominator
            )

        return score

    def retrieve(self, query: str) -> RetrievalResult:

        query_terms = self._tokenize(query)

        scored_documents = []

        for index in range(
            self.document_count
        ):

            score = self._score_document(
                query_terms,
                index,
            )

            scored_documents.append(
                (
                    index,
                    score,
                )
            )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        top_results = scored_documents[
            :self.top_k
        ]

        raw_scores = [
            score
            for _, score in top_results
        ]

        max_score = (
            max(raw_scores)
            if raw_scores
            else 1.0
        )

        retrieved_chunks = []

        for index, raw_score in top_results:

            if max_score > 0:

                normalized_score = (
                    raw_score
                    / max_score
                )

            else:

                normalized_score = 0.0

            document = self.documents[index]

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=document["id"],
                    text=document["text"],
                    score=normalized_score,
                    metadata={
                        **document.get(
                            "metadata",
                            {}
                        ),
                        "raw_bm25_score": raw_score,
                        "score_type": (
                            "normalized_bm25_relevance"
                        ),
                    },
                )
            )

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )