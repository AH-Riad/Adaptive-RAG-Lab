from dataclasses import dataclass


@dataclass
class RetrievalPlan:

    retrieval_strategy: str

    top_k: int

    chunk_strategy: str

    rerank: bool

    retry: bool

    confidence_threshold: float