from pathlib import Path
import pickle

import bm25s


class BM25SBenchmarkIndex:

    def __init__(
        self,
        dataset_name: str,
        output_dir: str = "datasets/processed"
    ):
        self.dataset_name = dataset_name

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index_dir = (
            self.output_dir
            / f"{dataset_name}_bm25s_index"
        )

        self.metadata_path = (
            self.output_dir
            / f"{dataset_name}_bm25s_metadata.pkl"
        )

        self.retriever = None
        self.document_ids = None

    def build(
        self,
        documents
    ):
        corpus = [
            document.text
            for document in documents
        ]

        self.document_ids = [
            document.id
            for document in documents
        ]

        print(
            "Tokenizing BM25 corpus..."
        )

        corpus_tokens = bm25s.tokenize(
            corpus
        )

        print(
            "Building BM25S index..."
        )

        self.retriever = bm25s.BM25(
            method="lucene"
        )

        self.retriever.index(
            corpus_tokens
        )

        self.retriever.save(
            str(self.index_dir)
        )

        metadata = {
            "dataset_name":
                self.dataset_name,

            "count":
                len(self.document_ids),

            "document_ids":
                self.document_ids
        }

        with self.metadata_path.open(
            "wb"
        ) as file:

            pickle.dump(
                metadata,
                file
            )

    def load(self):

        if not self.index_dir.exists():

            raise FileNotFoundError(
                f"BM25S index not found: "
                f"{self.index_dir}"
            )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                f"BM25S metadata not found: "
                f"{self.metadata_path}"
            )

        self.retriever = bm25s.BM25.load(
            str(self.index_dir),
            load_corpus=False,
            mmap=True
        )

        with self.metadata_path.open(
            "rb"
        ) as file:

            metadata = pickle.load(
                file
            )

        self.document_ids = (
            metadata["document_ids"]
        )

        return self

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        if self.retriever is None:

            raise RuntimeError(
                "BM25S index has not been loaded."
            )

        query_tokens = bm25s.tokenize(
            [query]
        )

        results, scores = (
            self.retriever.retrieve(
                query_tokens,
                k=top_k
            )
        )

        output = []

        for position in range(
            results.shape[1]
        ):

            document_index = int(
                results[0, position]
            )

            raw_score = float(
                scores[0, position]
            )

            output.append(
                {
                    "document_id":
                        self.document_ids[
                            document_index
                        ],

                    "score":
                        raw_score,

                    "index":
                        document_index
                }
            )

        return output