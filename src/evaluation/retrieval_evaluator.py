from dataclasses import dataclass, field


@dataclass
class RetrievalEvaluation:
    query: str
    query_type: str
    dense_ids: list[str] = field(default_factory=list)
    bm25_ids: list[str] = field(default_factory=list)
    hybrid_ids: list[str] = field(default_factory=list)


class RetrievalEvaluator:
    """
    Runs the same queries through multiple retrievers.
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        hybrid_retriever
    ):

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever

    def evaluate_query(
        self,
        query: str,
        query_type: str
    ) -> RetrievalEvaluation:

        dense_result = (
            self.dense_retriever.retrieve(query)
        )

        bm25_result = (
            self.bm25_retriever.retrieve(query)
        )

        hybrid_result = (
            self.hybrid_retriever.retrieve(query)
        )

        dense_ids = [
            chunk.chunk_id
            for chunk in dense_result.retrieved_chunks
        ]

        bm25_ids = [
            chunk.chunk_id
            for chunk in bm25_result.retrieved_chunks
        ]

        hybrid_ids = [
            chunk.chunk_id
            for chunk in hybrid_result.retrieved_chunks
        ]

        return RetrievalEvaluation(
            query=query,
            query_type=query_type,
            dense_ids=dense_ids,
            bm25_ids=bm25_ids,
            hybrid_ids=hybrid_ids
        )

    def evaluate(
        self,
        queries: list[dict]
    ) -> list[RetrievalEvaluation]:

        results = []

        for item in queries:

            result = self.evaluate_query(
                query=item["query"],
                query_type=item["type"]
            )

            results.append(result)

        return results