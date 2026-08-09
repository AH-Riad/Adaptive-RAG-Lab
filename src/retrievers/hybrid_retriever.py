from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.core import RetrievedChunk


class HybridRetriever(BaseRetriever):
    """
    Combines dense semantic retrieval with BM25
    lexical retrieval.

    Hybrid score:

        score =
            alpha * dense_score
            +
            (1 - alpha) * bm25_score
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        top_k: int = 3,
        alpha: float = 0.7,
    ):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0.0 and 1.0"
            )

        self.dense_retriever = (
            dense_retriever
        )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.top_k = top_k
        self.alpha = alpha

    # RETRIEVE

    def retrieve(
        self,
        query: str
    ) -> RetrievalResult:

        # Dense retrieval

        dense_result = (
            self.dense_retriever.retrieve(
                query
            )
        )

        # BM25 retrieval

        bm25_result = (
            self.bm25_retriever.retrieve(
                query
            )
        )

        # Build lookup tables

        dense_chunks = {
            chunk.chunk_id: chunk
            for chunk
            in dense_result.retrieved_chunks
        }

        bm25_chunks = {
            chunk.chunk_id: chunk
            for chunk
            in bm25_result.retrieved_chunks
        }

        all_ids = (
            set(dense_chunks.keys())
            |
            set(bm25_chunks.keys())
        )

        # Calculate hybrid scores

        ranked_chunks = []

        for chunk_id in all_ids:

            dense_score = 0.0
            bm25_score = 0.0

            if chunk_id in dense_chunks:

                dense_score = (
                    dense_chunks[
                        chunk_id
                    ].score
                )

            if chunk_id in bm25_chunks:

                bm25_score = (
                    bm25_chunks[
                        chunk_id
                    ].score
                )

            hybrid_score = (
                self.alpha * dense_score
                +
                (1.0 - self.alpha)
                * bm25_score
            )

            # Pick the original chunk object
            # from whichever retriever returned it.
            if chunk_id in dense_chunks:

                base_chunk = (
                    dense_chunks[
                        chunk_id
                    ]
                )

            else:

                base_chunk = (
                    bm25_chunks[
                        chunk_id
                    ]
                )

            ranked_chunks.append(
                (
                    base_chunk,
                    dense_score,
                    bm25_score,
                    hybrid_score
                )
            )

        # Rank by hybrid score

        ranked_chunks.sort(
            key=lambda item: item[3],
            reverse=True
        )

        ranked_chunks = ranked_chunks[
            :self.top_k
        ]

        # Build final RetrievalResult

        retrieved_chunks = []

        for (
            chunk,
            dense_score,
            bm25_score,
            hybrid_score
        ) in ranked_chunks:

            metadata = dict(
                chunk.metadata
            )

            metadata.update({

                "dense_score":
                    dense_score,

                "bm25_score":
                    bm25_score,

                "hybrid_score":
                    hybrid_score,

                "fusion_alpha":
                    self.alpha,

                "score_type":
                    "hybrid_relevance",
            })

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=hybrid_score,
                    metadata=metadata
                )
            )

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks
        )