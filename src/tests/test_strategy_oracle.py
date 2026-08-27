import json

from src.evaluation.strategy_oracle import (
    StrategyOracle
)


def main():

    print("=" * 60)
    print("D²RAG STRATEGY ORACLE ANALYSIS")
    print("=" * 60)

    oracle = StrategyOracle(
        results_path=(
            "results/logs/"
            "fiqa_full_test_results.json"
        )
    )

    report = oracle.calculate()

    print(
        "\nEvaluated Queries:",
        report["queries"]
    )

    print(
        "Strategy Selection Accuracy:",
        round(
            report[
                "strategy_selection_accuracy"
            ],
            4
        )
    )

    print(
        "Adaptation Rate:",
        round(
            report["adaptation_rate"],
            4
        )
    )

    print(
        "Strategy Transition Rate:",
        round(
            report[
                "strategy_transition_rate"
            ],
            4
        )
    )

    print(
        "Successful Recovery Rate:",
        round(
            report[
                "successful_recovery_rate"
            ],
            4
        )
    )

    print(
        "Failed Adaptation Rate:",
        round(
            report[
                "failed_adaptation_rate"
            ],
            4
        )
    )

    print(
        "Average Oracle Regret:",
        round(
            report[
                "average_oracle_regret"
            ],
            4
        )
    )

    output_path = (
        "results/logs/"
        "fiqa_strategy_oracle.json"
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
        "\nOracle report saved to:"
    )

    print(output_path)

    print("\n" + "=" * 60)
    print(
        "STRATEGY ORACLE ANALYSIS COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()