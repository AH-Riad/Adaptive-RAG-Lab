from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AdaptiveContext:
    """
    The central state object passed through the pipeline.
    No component owns the data; the pipeline owns the state.
    """
    query: str
    query_analysis: Optional[Dict[str, Any]] = None
    retrieval_plan: Optional[Dict[str, Any]] = None
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    confidence_report: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)