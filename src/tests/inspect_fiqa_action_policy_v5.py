import json
from collections import Counter


POLICY_PATH = (
    "results/logs/"
    "fiqa_dev_action_policy_v5.json"
)


def load_policy():

    with open(
        POLICY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def inspect_metadata(
    policy
):

    print("\n" + "=" * 60)
    print("POLICY METADATA")
    print("=" * 60)

    print(
        "Objective:",
        policy.get(
            "objective",
            {}
        )
    )

    print(
        "Supported Top-K:",
        policy.get(
            "supported_top_k",
            []
        )
    )

    print(
        "Top-K candidate rule:",
        policy.get(
            "topk_candidate_rule",
            "missing"
        )
    )

    print(
        "Strategy states:",
        len(
            policy.get(
                "strategy_policy",
                {}
            )
        )
    )

    print(
        "Top-K states:",
        len(
            policy.get(
                "topk_policy",
                {}
            )
        )
    )


def inspect_strategy_policy(
    policy
):

    strategy_policy = (
        policy.get(
            "strategy_policy",
            {}
        )
    )

    print("\n" + "=" * 60)
    print("STRATEGY ACTION POLICY")
    print("=" * 60)

    action_counts = Counter()

    for state, data in (
        strategy_policy.items()
    ):

        action = data.get(
            "selected_action",
            "unknown"
        )

        samples = data.get(
            "samples",
            0
        )

        action_counts[action] += 1

        print(
            f"State: {state}\n"
            f"  Selected action: {action}\n"
            f"  Samples: {samples}"
        )

    print("\nStrategy action distribution:")

    for action, count in (
        action_counts.items()
    ):

        print(
            f"  {action}: {count}"
        )


def parse_current_k(
    state
):

    parts = state.strip(
        "()"
    ).split(", ")

    try:

        return int(
            parts[-1]
            .strip("'")
        )

    except (
        ValueError,
        IndexError
    ):

        return None


def inspect_topk_policy(
    policy
):

    topk_policy = (
        policy.get(
            "topk_policy",
            {}
        )
    )

    print("\n" + "=" * 60)
    print("TOP-K ACTION POLICY")
    print("=" * 60)

    action_counts = Counter()

    invalid = []

    for state, data in (
        topk_policy.items()
    ):

        action = data.get(
            "selected_action",
            "unknown"
        )

        samples = data.get(
            "samples",
            0
        )

        action_counts[action] += 1

        current_k = parse_current_k(
            state
        )

        if (
            action.startswith(
                "set_top_k_"
            )
            and current_k is not None
        ):

            target_k = int(
                action.split(
                    "_"
                )[-1]
            )

            if target_k <= current_k:

                invalid.append(
                    (
                        state,
                        action
                    )
                )

        print(
            f"State: {state}\n"
            f"  Selected action: {action}\n"
            f"  Samples: {samples}"
        )

    print("\nTop-K action distribution:")

    for action, count in (
        action_counts.items()
    ):

        print(
            f"  {action}: {count}"
        )

    print("\nTop-K transition validation:")

    if invalid:

        print(
            "  INVALID transitions found:"
        )

        for state, action in invalid:

            print(
                f"    {state} -> {action}"
            )

    else:

        print(
            "  All selected Top-K "
            "changes are valid expansions."
        )


def inspect_topk_candidates(
    policy
):

    topk_policy = (
        policy.get(
            "topk_policy",
            {}
        )
    )

    print("\n" + "=" * 60)
    print("TOP-K CANDIDATE CHECK")
    print("=" * 60)

    for state, data in (
        topk_policy.items()
    ):

        current_k = parse_current_k(
            state
        )

        print(
            f"\nState: {state}"
        )

        print(
            f"Current K: {current_k}"
        )

        candidates = data.get(
            "candidates",
            {}
        )

        for action, candidate in (
            candidates.items()
        ):

            print(
                f"  {action}: "
                f"samples="
                f"{candidate.get('count', 0)}, "
                f"utility="
                f"{candidate.get('average_utility')}"
            )


def main():

    print("=" * 60)
    print(
        "FIQA DEVELOPMENT ACTION POLICY V5 INSPECTION"
    )
    print("=" * 60)

    policy = load_policy()

    inspect_metadata(
        policy
    )

    inspect_strategy_policy(
        policy
    )

    inspect_topk_policy(
        policy
    )

    inspect_topk_candidates(
        policy
    )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()