from src.core.component import Component
from src.retrievers.retrieval_result import RetrievalResult
from src.planning.decision_types import RetrievalStrategy


class AdaptiveRetriever(Component):
    """
    Executes the retrieval strategy selected by the
    Decision Engine.
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        hybrid_retriever
    ):
        self.retrievers = {
            RetrievalStrategy.DENSE:
                dense_retriever,

            RetrievalStrategy.BM25:
                bm25_retriever,

            RetrievalStrategy.HYBRID:
                hybrid_retriever,
        }

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int
    ) -> RetrievalResult:

        if strategy not in self.retrievers:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy}"
            )

        retriever = self.retrievers[strategy]

        if hasattr(retriever, "top_k"):
            retriever.top_k = top_k

        result = retriever.retrieve(query)

        return result

    def run(self, context):

        plan = context.retrieval_plan

        if plan is None:
            raise RuntimeError(
                "Retrieval plan is required."
            )

        result = self.retrieve(
            query=context.query,
            strategy=plan.strategy,
            top_k=plan.top_k
        )

        context.retrieval_result = result

        context.add_event(
            f"retrieval_executed:{plan.strategy.value}"
        )

        return context