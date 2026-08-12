from dataclasses import dataclass


@dataclass
class EvaluationResult:

    query: str
    query_type: str
    retriever: str

    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float