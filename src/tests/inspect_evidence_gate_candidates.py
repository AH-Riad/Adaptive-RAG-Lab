import json
from pathlib import Path


INPUT_PATH = Path(
    "results/logs/"
    "fiqa_dev_evidence_gate_sweep_v1.json"
)


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        artifact = json.load(file)

    candidates = (
        artifact["top_practical_candidates"]
    )

    all_candidates = []

    for result in artifact.get(
        "top_by_f1",
        []
    ):
        all_candidates.append(result)

    for result in artifact.get(
        "top_by_precision",
        []
    ):
        all_candidates.append(result)

    for result in candidates:
        all_candidates.append(result)

    unique = {}

    for result in all_candidates:
        key = (
            result["confidence_threshold"],
            result["coverage_threshold"],
            result["gap_threshold"]
        )

        unique[key] = result

    filtered = [
        result
        for result in unique.values()
        if result["false_accept_rate"] <= 0.20
        and 0.25 <= result["acceptance_rate"] <= 0.75
    ]

    filtered.sort(
        key=lambda result: (
            result["acceptance_recall"],
            result["acceptance_precision"],
            result["f1"]
        ),
        reverse=True
    )

    print(
        "Candidates with false accept rate <= 0.20:"
    )

    if not filtered:
        print(
            "  none found in the inspected candidates"
        )

    for number, result in enumerate(
        filtered[:10],
        start=1
    ):
        print(
            f"{number}. "
            f"confidence>={result['confidence_threshold']:.2f}, "
            f"coverage>={result['coverage_threshold']:.2f}, "
            f"gap>={result['gap_threshold']:.2f} | "
            f"acceptance_rate={result['acceptance_rate']:.3f}, "
            f"precision={result['acceptance_precision']:.3f}, "
            f"recall={result['acceptance_recall']:.3f}, "
            f"false_accept={result['false_accept_rate']:.3f}, "
            f"false_reject={result['false_reject_rate']:.3f}, "
            f"F1={result['f1']:.3f}"
        )


if __name__ == "__main__":
    main()