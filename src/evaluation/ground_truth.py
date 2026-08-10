from dataclasses import dataclass, field


@dataclass
class GroundTruthQuery:
    query: str
    query_type: str
    relevant_chunks: list[str] = field(
        default_factory=list
    )


class GroundTruthDataset:
    """
    Stores manually verified relevant chunks for queries.
    """

    def __init__(self):

        self.queries = []

    def add(
        self,
        query: str,
        query_type: str,
        relevant_chunks: list[str]
    ):

        self.queries.append(
            GroundTruthQuery(
                query=query,
                query_type=query_type,
                relevant_chunks=relevant_chunks
            )
        )

    def get_all(self):
        return self.queries

    def get(
        self,
        query: str
    ):

        for item in self.queries:

            if item.query == query:
                return item

        return None