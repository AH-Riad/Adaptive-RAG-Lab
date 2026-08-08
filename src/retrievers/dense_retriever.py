from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.retrieval_result import RetrievalResult
from src.retrievers.score_normalizer import ScoreNormalizer
from src.core import RetrievedChunk


class DenseRetriever(BaseRetriever):

    def __init__(
        self,
        embedding_model,
        vector_store,
        top_k: int = 3
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:

        query_embedding = (
            self.embedding_model.encode_query(query)
        )

        raw_results = self.vector_store.search(
            query_embedding,
            top_k=self.top_k
        )

        retrieved_chunks = []

        if (
            raw_results
            and "ids" in raw_results
            and len(raw_results["ids"]) > 0
        ):

            ids = raw_results["ids"][0]

            distances = (
                raw_results["distances"][0]
                if "distances" in raw_results
                else [0.0] * len(ids)
            )

            documents = (
                raw_results["documents"][0]
                if "documents" in raw_results
                else [""] * len(ids)
            )

            metadatas = (
                raw_results["metadatas"][0]
                if "metadatas" in raw_results
                else [{}] * len(ids)
            )

            for i in range(len(ids)):

                distance = float(
                    distances[i]
                )

                relevance_score = (
                    ScoreNormalizer.distance_to_relevance(
                        distance
                    )
                )

                chunk = RetrievedChunk(

                    chunk_id=ids[i],

                    text=documents[i],

                    score=relevance_score,

                    metadata={
                        **metadatas[i],
                        "raw_distance": distance,
                        "score_type": "normalized_relevance",
                    }
                )

                retrieved_chunks.append(chunk)

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks
        )