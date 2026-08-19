from src.core import RetrievedChunk
from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.retrievers.score_fusion import WeightedScoreFusion


class BenchmarkHybridRetriever(BaseRetriever):

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        top_k: int = 5,
        alpha: float = 0.7
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

        self.fusion = WeightedScoreFusion(
            alpha=alpha
        )

    def retrieve(
        self,
        query: str
    ) -> RetrievalResult:

        dense_result = (
            self.dense_retriever.retrieve(
                query
            )
        )

        bm25_result = (
            self.bm25_retriever.retrieve(
                query
            )
        )

        dense_chunks = {
            chunk.chunk_id: chunk
            for chunk in
            dense_result.retrieved_chunks
        }

        bm25_chunks = {
            chunk.chunk_id: chunk
            for chunk in
            bm25_result.retrieved_chunks
        }

        dense_scores = {
            chunk_id: chunk.score
            for chunk_id, chunk
            in dense_chunks.items()
        }

        bm25_scores = {
            chunk_id: chunk.score
            for chunk_id, chunk
            in bm25_chunks.items()
        }

        fused = self.fusion.fuse(
            dense_scores=dense_scores,
            bm25_scores=bm25_scores
        )

        fused = [
            result
            for result in fused
            if result.hybrid_score > 0
        ]

        fused = fused[:self.top_k]

        retrieved_chunks = []

        for result in fused:

            if result.chunk_id in dense_chunks:

                base_chunk = dense_chunks[
                    result.chunk_id
                ]

            else:

                base_chunk = bm25_chunks[
                    result.chunk_id
                ]

            metadata = dict(
                base_chunk.metadata
            )

            metadata.update({
                "dense_score":
                    result.dense_score,

                "bm25_score":
                    result.bm25_score,

                "hybrid_score":
                    result.hybrid_score,

                "fusion_alpha":
                    self.fusion.alpha,

                "score_type":
                    "benchmark_hybrid_relevance"
            })

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text=base_chunk.text,
                    score=result.hybrid_score,
                    metadata=metadata
                )
            )

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks
        )