from dataclasses import dataclass, field


@dataclass
class GroundTruthQuery:
    query: str
    query_type: str
    relevant_chunks: list[str] = field(
        default_factory=list
    )
    relevance_scores: dict[str, int] = field(
        default_factory=dict
    )


class GroundTruthDataset:
    """
    Stores manually verified relevance judgments.
    """

    def __init__(self):

        self.queries = []

    def add(
        self,
        query: str,
        query_type: str,
        relevant_chunks: list[str],
        relevance_scores: dict[str, int] | None = None
    ):

        if relevance_scores is None:
            relevance_scores = {
                chunk_id: 1
                for chunk_id in relevant_chunks
            }

        self.queries.append(
            GroundTruthQuery(
                query=query,
                query_type=query_type,
                relevant_chunks=relevant_chunks,
                relevance_scores=relevance_scores
            )
        )

    def get_all(self):
        return self.queries

    def get(self, query: str):

        for item in self.queries:

            if item.query == query:
                return item

        return None