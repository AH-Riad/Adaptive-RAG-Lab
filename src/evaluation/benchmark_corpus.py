from src.core import Document


class BenchmarkCorpus:

    def __init__(
        self,
        dataset_name: str,
        corpus: dict
    ):

        self.dataset_name = dataset_name
        self.corpus = corpus

    def to_documents(self):

        documents = []

        for document_id, record in (
            self.corpus.items()
        ):

            title = record.get(
                "title",
                ""
            )

            text = record.get(
                "text",
                ""
            )

            if title:

                combined_text = (
                    f"{title}\n{text}"
                )

            else:

                combined_text = text

            documents.append(
                Document(
                    id=str(document_id),
                    source=(
                        f"BEIR:{self.dataset_name}"
                    ),
                    text=combined_text,
                    metadata={
                        "benchmark": "BEIR",
                        "dataset":
                            self.dataset_name,
                        "benchmark_id":
                            str(document_id)
                    }
                )
            )

        return documents