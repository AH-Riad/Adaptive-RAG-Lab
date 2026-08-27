import json
from pathlib import Path

from src.evaluation.strategy_calibrator import (
    StrategyCalibrator
)


def save_calibration(
    calibration,
    dataset_name: str,
    split: str = "dev",
    version: str = "v1"
):

    output_dir = Path(
        "results"
        "/"
        "logs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        output_dir
        /
        f"{dataset_name}_"
        f"{split}_strategy_policy_"
        f"{version}.json"
    )

    artifact = {
        "dataset":
            dataset_name,

        "split":
            split,

        "policy_version":
            version,

        "policy":
            calibration["policy"],

        "report":
            calibration["report"]
    }

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            artifact,
            file,
            indent=2
        )

    return path