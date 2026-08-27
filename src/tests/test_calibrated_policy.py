from src.planning.calibrated_policy import (
    CalibratedPolicy
)


def main():

    print("=" * 60)
    print("CALIBRATED POLICY TEST")
    print("=" * 60)

    policy = CalibratedPolicy(
        path=(
            "results/logs/"
            "fiqa_dev_strategy_policy_v1.json"
        )
    )

    print(
        "Dataset:",
        policy.dataset
    )

    print(
        "Calibration split:",
        policy.split
    )

    print(
        "Policy version:",
        policy.version
    )

    print("\nFrozen strategy mapping:")

    for query_type in [
        "ambiguous",
        "lexical",
        "semantic",
        "comparison"
    ]:

        strategy = policy.get_strategy(
            query_type
        )

        print(
            f"{query_type} -> "
            f"{strategy.value}"
        )

    assert (
        policy.get_strategy(
            "ambiguous"
        ).value
        == "dense"
    )

    assert (
        policy.get_strategy(
            "lexical"
        ).value
        == "dense"
    )

    assert (
        policy.get_strategy(
            "semantic"
        ).value
        == "hybrid"
    )

    assert (
        policy.get_strategy(
            "comparison"
        ).value
        == "hybrid"
    )

    print("\n" + "=" * 60)
    print(
        "CALIBRATED POLICY TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()