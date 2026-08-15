from dataclasses import dataclass


@dataclass
class BenchmarkQuery:

    query_id: str
    query: str
    relevant_documents: dict[str, int]


@dataclass
class BenchmarkDataset:

    name: str
    corpus: dict
    queries: dict
    qrels: dict

    def get_queries(self):

        results = []

        for query_id, query in self.queries.items():

            relevant = self.qrels.get(
                query_id,
                {}
            )

            results.append(
                BenchmarkQuery(
                    query_id=query_id,
                    query=query,
                    relevant_documents=relevant
                )
            )

        return results

    def get_corpus_size(self):

        return len(self.corpus)

    def get_query_count(self):

        return len(self.queries)