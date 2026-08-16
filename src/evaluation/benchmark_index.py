from pathlib import Path

import pickle


class BenchmarkIndex:

    def __init__(
        self,
        dataset_name: str,
        index_dir: str = "datasets/processed"
    ):

        self.dataset_name = dataset_name

        self.index_dir = Path(
            index_dir
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    @property
    def metadata_path(self):

        return (
            self.index_dir
            / f"{self.dataset_name}_metadata.pkl"
        )

    def save_metadata(
        self,
        metadata
    ):

        with self.metadata_path.open(
            "wb"
        ) as file:

            pickle.dump(
                metadata,
                file
            )

    def load_metadata(self):

        if not self.metadata_path.exists():
            return None

        with self.metadata_path.open(
            "rb"
        ) as file:

            return pickle.load(
                file
            )