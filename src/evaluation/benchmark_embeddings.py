from pathlib import Path
import pickle

import numpy as np


class BenchmarkEmbeddingStore:

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        output_dir: str = "datasets/processed"
    ):

        self.dataset_name = dataset_name
        self.model_name = model_name

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_model_name = (
            model_name
            .replace("/", "_")
            .replace("\\", "_")
        )

        self.path = (
            self.output_dir
            /
            (
                f"{dataset_name}_"
                f"{safe_model_name}_"
                "embeddings.npy"
            )
        )

        self.metadata_path = (
            self.output_dir
            /
            (
                f"{dataset_name}_"
                f"{safe_model_name}_"
                "embedding_metadata.pkl"
            )
        )

    def save(
        self,
        embeddings,
        document_ids: list[str]
    ):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        np.save(
            self.path,
            embeddings
        )

        metadata = {
            "dataset_name":
                self.dataset_name,

            "model_name":
                self.model_name,

            "count":
                len(document_ids),

            "embedding_dimension":
                (
                    embeddings.shape[1]
                    if embeddings.ndim == 2
                    else 0
                ),

            "document_ids":
                document_ids
        }

        with self.metadata_path.open(
            "wb"
        ) as file:

            pickle.dump(
                metadata,
                file
            )

    def load(self):

        embeddings = np.load(
            self.path,
            mmap_mode="r"
        )

        with self.metadata_path.open(
            "rb"
        ) as file:

            metadata = pickle.load(
                file
            )

        return embeddings, metadata

    def exists(self):

        return (
            self.path.exists()
            and
            self.metadata_path.exists()
        )