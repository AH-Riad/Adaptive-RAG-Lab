import json
from pathlib import Path

from src.evaluation.experiment_record import (
    ExperimentRecord
)


class ExperimentTracker:

    def __init__(
        self,
        output_path: str = "results/logs/experiments.jsonl"
    ):

        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def record(
        self,
        experiment: ExperimentRecord
    ):

        with self.output_path.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    experiment.to_dict()
                )
                + "\n"
            )

    def load_all(self):

        if not self.output_path.exists():
            return []

        records = []

        with self.output_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if line:
                    records.append(
                        json.loads(line)
                    )

        return records

    def count(self):

        return len(
            self.load_all()
        )