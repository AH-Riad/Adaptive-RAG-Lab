import re
import math
from collections import Counter

from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.core import RetrievedChunk


class BM25Retriever(BaseRetriever):
    """
    Lexical retrieval using BM25.

    BM25 is useful for:
    - exact keywords
    - technical terms
    - names
    - identifiers
    - lexical matching
    """

    def __init__(
        self,
        chunks,
        top_k: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = chunks
        self.top_k = top_k
        self.k1 = k1
        self.b = b

        self._build_index()

    # TOKENIZATION

    def _tokenize(self, text: str) -> list[str]:
        """
        Normalize text and tokenize it.

        Punctuation is removed so that:

            self-attention.
            self-attention,

        are treated as the same lexical term.
        """

        text = text.lower()

        tokens = re.findall(
            r"\b[\w]+(?:-[\w]+)*\b",
            text
        )

        return tokens

    # BUILD INDEX

    def _build_index(self):

        self.tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.document_count = len(
            self.tokenized_chunks
        )

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_chunks
        ]

        if self.document_count > 0:

            self.average_document_length = (
                sum(self.document_lengths)
                / self.document_count
            )

        else:

            self.average_document_length = 0.0

        # Number of chunks containing each term
        self.document_frequency = Counter()

        for tokens in self.tokenized_chunks:

            unique_terms = set(tokens)

            for term in unique_terms:

                self.document_frequency[
                    term
                ] += 1

    # IDF

    def _idf(self, term: str) -> float:

        df = self.document_frequency.get(
            term,
            0
        )

        if df == 0:
            return 0.0

        return math.log(
            1
            +
            (
                self.document_count
                - df
                + 0.5
            )
            /
            (
                df
                + 0.5
            )
        )

    # SCORE ONE CHUNK

    def _score_chunk(
        self,
        query_terms: list[str],
        chunk_index: int,
    ) -> float:

        tokens = self.tokenized_chunks[
            chunk_index
        ]

        if not tokens:
            return 0.0

        term_frequency = Counter(tokens)

        document_length = (
            self.document_lengths[
                chunk_index
            ]
        )

        score = 0.0

        for term in query_terms:

            frequency = term_frequency.get(
                term,
                0
            )

            if frequency == 0:
                continue

            idf = self._idf(term)

            numerator = (
                frequency
                * (self.k1 + 1)
            )

            denominator = (
                frequency
                +
                self.k1
                *
                (
                    1
                    - self.b
                    +
                    self.b
                    *
                    (
                        document_length
                        /
                        self.average_document_length
                    )
                )
            )

            score += (
                idf
                * numerator
                / denominator
            )

        return score

    # RETRIEVE

    def retrieve(
        self,
        query: str
    ) -> RetrievalResult:

        query_terms = self._tokenize(
            query
        )

        scored_chunks = []

        for index in range(
            self.document_count
        ):

            score = self._score_chunk(
                query_terms,
                index
            )

            scored_chunks.append(
                (
                    index,
                    score
                )
            )

        # Highest BM25 score first
        scored_chunks.sort(
            key=lambda item: item[1],
            reverse=True
        )

        top_results = scored_chunks[
            :self.top_k
        ]

        # Normalize scores

        raw_scores = [
            score
            for _, score in top_results
        ]

        max_score = (
            max(raw_scores)
            if raw_scores
            else 0.0
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

            chunk = self.chunks[index]

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=normalized_score,
                    metadata={
                        **chunk.metadata,
                        "raw_bm25_score": raw_score,
                        "score_type":
                            "normalized_bm25_relevance",
                    }
                )
            )

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks
        )