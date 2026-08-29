import json

from src.evaluation.evidence_calibration import (
    EvidenceCalibrationAnalyzer
)


def main():

    print("=" * 60)
    print("D²RAG EVIDENCE CONFIDENCE CALIBRATION")
    print("=" * 60)

    analyzer = EvidenceCalibrationAnalyzer(
        results_path=(
            "results/logs/"
            "fiqa_full_test_results.json"
        ),
        number_of_bins=10
    )

    report = analyzer.analyze()

    print(
        "\nQueries:",
        report["queries"]
    )

    print(
        "Brier Score:",
        round(
            report["brier_score"],
            4
        )
    )

    print(
        "Expected Calibration Error:",
        round(
            report[
                "expected_calibration_error"
            ],
            4
        )
    )

    acceptance = report[
        "acceptance"
    ]

    print(
        "\nAcceptance Precision:",
        round(
            acceptance[
                "acceptance_precision"
            ],
            4
        )
    )

    print(
        "Acceptance Recall:",
        round(
            acceptance[
                "acceptance_recall"
            ],
            4
        )
    )

    print(
        "False Acceptance Rate:",
        round(
            acceptance[
                "false_acceptance_rate"
            ],
            4
        )
    )

    confidence = report[
        "confidence"
    ]

    print(
        "\nMean Confidence "
        "When Relevant:",
        round(
            confidence[
                "mean_confidence_when_relevant"
            ],
            4
        )
    )

    print(
        "Mean Confidence "
        "When Irrelevant:",
        round(
            confidence[
                "mean_confidence_when_irrelevant"
            ],
            4
        )
    )

    output_path = (
        "results/logs/"
        "fiqa_evidence_calibration.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print(
        "\nCalibration report saved to:"
    )

    print(output_path)

    print("\n" + "=" * 60)
    print(
        "EVIDENCE CALIBRATION ANALYSIS COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()