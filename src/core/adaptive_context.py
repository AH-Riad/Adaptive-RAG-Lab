from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AdaptiveContext:
    """
    Shared state passed through the entire Adaptive RAG pipeline.

    Every component reads from and writes to this object.
    """

    # Query
    query: str
    query_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Analysis
    query_analysis: dict[str, Any] = field(default_factory=dict)

    # Planning
    retrieval_plan: dict[str, Any] = field(default_factory=dict)

    # Retrieval
    retrieved_documents: list[Any] = field(default_factory=list)

    # Assessment
    assessment: dict[str, Any] = field(default_factory=dict)

    # Generation
    generated_answer: str | None = None

    # Experiment
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    # History
    events: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    analysis_confidence: dict = field(default_factory=dict)

    decision_report: dict = field(default_factory=dict)

    metrics: dict = field(default_factory=dict)
    
    def add_event(self, event: str) -> None:
        self.events.append(event)

    def add_log(self, message: str) -> None:
        self.logs.append(message)
    
