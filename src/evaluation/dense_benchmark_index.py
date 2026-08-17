from pathlib import Path

import numpy as np


class DenseBenchmarkIndex:

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str
    ):
        self.embeddings_path = Path(
            embeddings_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.embeddings = None
        self.metadata = None
        self.document_ids = None

    def load(self):

        if not self.embeddings_path.exists():

            raise FileNotFoundError(
                f"Embeddings not found: "
                f"{self.embeddings_path}"
            )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata not found: "
                f"{self.metadata_path}"
            )

        self.embeddings = np.load(
            self.embeddings_path,
            mmap_mode="r"
        )

        import pickle

        with self.metadata_path.open(
            "rb"
        ) as file:

            self.metadata = pickle.load(
                file
            )

        self.document_ids = (
            self.metadata[
                "document_ids"
            ]
        )

        if len(self.document_ids) != (
            self.embeddings.shape[0]
        ):

            raise ValueError(
                "Embedding rows and document IDs "
                "are not aligned."
            )

        return self

    def search(
        self,
        query_embedding,
        top_k: int = 5
    ):

        if self.embeddings is None:

            raise RuntimeError(
                "Index has not been loaded."
            )

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_vector.ndim != 1:

            raise ValueError(
                "Query embedding must be 1-dimensional."
            )

        if (
            query_vector.shape[0]
            != self.embeddings.shape[1]
        ):

            raise ValueError(
                "Query embedding dimension does "
                "not match the index."
            )

        query_norm = np.linalg.norm(
            query_vector
        )

        if query_norm == 0:

            raise ValueError(
                "Query embedding cannot be zero."
            )

        query_vector = (
            query_vector
            / query_norm
        )

        scores = (
            self.embeddings
            @ query_vector
        )

        top_k = min(
            top_k,
            len(self.document_ids)
        )

        indices = np.argpartition(
            -scores,
            top_k - 1
        )[:top_k]

        indices = indices[
            np.argsort(
                -scores[indices]
            )
        ]

        results = []

        for index in indices:

            index = int(index)

            results.append(
                {
                    "document_id":
                        self.document_ids[index],

                    "score":
                        float(scores[index]),

                    "index":
                        index
                }
            )

        return results