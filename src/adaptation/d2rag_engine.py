from src.core.component import Component
from src.adaptation.adaptive_retrieval_orchestrator import (
    AdaptiveRetrievalOrchestrator
)


class D2RAGEngine(Component):
    """
    End-to-end D²RAG retrieval engine.

    Flow:

    Query
    ↓
    Query Analysis
    ↓
    Pre-Retrieval Decision
    ↓
    Retrieval
    ↓
    Evidence Assessment
    ↓
    Post-Retrieval Decision
    ↓
    Retry if necessary
    """

    def __init__(
        self,
        query_analyzer,
        adaptive_retrieval_orchestrator
    ):
        self.query_analyzer = query_analyzer
        self.orchestrator = (
            adaptive_retrieval_orchestrator
        )

    def run(self, context):

        analysis = self.query_analyzer.analyze(
            context.query
        )

        context.query_analysis = analysis

        context.add_event(
            "query_analysis_completed"
        )

        context = self.orchestrator.run(
            context
        )

        context.add_event(
            "d2rag_execution_completed"
        )

        return context