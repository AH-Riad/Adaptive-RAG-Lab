from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.evaluation_result import (
    EvaluationResult
)


class RetrievalEvaluator:
    """
    Evaluates a retriever against manually verified
    ground-truth relevance judgments.
    """

    def evaluate(
        self,
        retriever,
        ground_truth
    ):

        results = []

        for item in ground_truth.get_all():

            retrieval_result = retriever.retrieve(
                item.query
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk in retrieval_result.retrieved_chunks
            ]

            precision = (
                RetrievalMetrics.precision_at_k(
                    retrieved_ids,
                    item.relevant_chunks,
                    5
                )
            )

            recall = (
                RetrievalMetrics.recall_at_k(
                    retrieved_ids,
                    item.relevant_chunks,
                    5
                )
            )

            mrr = (
                RetrievalMetrics.reciprocal_rank(
                    retrieved_ids,
                    item.relevant_chunks
                )
            )

            ndcg = (
                RetrievalMetrics.ndcg_at_k(
                    retrieved_ids,
                    item.relevance_scores,
                    5
                )
            )

            results.append(
                EvaluationResult(
                    query=item.query,
                    query_type=item.query_type,
                    retriever=retriever.__class__.__name__,
                    precision_at_5=precision,
                    recall_at_5=recall,
                    mrr=mrr,
                    ndcg_at_5=ndcg
                )
            )

        return results