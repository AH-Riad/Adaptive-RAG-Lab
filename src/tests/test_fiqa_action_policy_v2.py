import json


def main():

    print("=" * 60)
    print("FIQA ACTION POLICY V2 INSPECTION")
    print("=" * 60)

    path = (
        "results/logs/"
        "fiqa_dev_action_policy_v2.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        artifact = json.load(file)

    print(
        "Dataset:",
        artifact["dataset"]
    )

    print(
        "Split:",
        artifact["split"]
    )

    print(
        "Version:",
        artifact["version"]
    )

    print(
        "\nStrategy policy:"
    )

    for state, value in (
        artifact[
            "strategy_policy"
        ].items()
    ):

        print(
            state,
            "->",
            value[
                "selected_action"
            ],
            "| samples:",
            value[
                "samples"
            ]
        )

    print(
        "\nTop-K policy:"
    )

    for state, value in (
        artifact[
            "topk_policy"
        ].items()
    ):

        print(
            state,
            "->",
            value[
                "selected_action"
            ],
            "| samples:",
            value[
                "samples"
            ]
        )

    assert (
        artifact["dataset"]
        == "fiqa"
    )

    assert (
        artifact["split"]
        == "dev"
    )

    assert len(
        artifact["strategy_policy"]
    ) > 0

    assert len(
        artifact["topk_policy"]
    ) > 0

    print("\n" + "=" * 60)
    print(
        "FIQA ACTION POLICY V2 INSPECTION PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()