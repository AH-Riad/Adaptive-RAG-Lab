from dataclasses import dataclass


@dataclass
class RetrievalPlan:
    retrieval_strategy: str
    top_k: int
    chunk_strategy: str
    use_reranking: bool
    retry_on_low_confidence: bool