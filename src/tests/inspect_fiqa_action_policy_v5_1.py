import json

POLICY_PATH = (
    "results/logs/"
    "fiqa_dev_action_policy_v5_1.json"
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
    print("\n" + "=" * 60)
    print("STRATEGY POLICY")
    print("=" * 60)

    strategy_policy = (
        policy.get(
            "strategy_policy",
            {}
        )
    )

    for state, data in (
        strategy_policy.items()
    ):

        print(
            f"State: {state}"
        )

        print(
            "  Action:",
            data.get(
                "selected_action"
            )
        )

        print(
            "  Query count:",
            data.get(
                "query_count",
                0
            )
        )

        print(
            "  Candidates:"
        )

        for action, candidate in (
            data.get(
                "candidates",
                {}
            ).items()
        ):

            print(
                f"    {action}: "
                f"{candidate.get('average_utility', 0.0):.6f}"
            )


def parse_state(
    state
):
    parts = state.strip(
        "()"
    ).split(", ")

    return parts


def current_k(
    state
):
    parts = parse_state(
        state
    )

    try:

        return int(
            parts[-1].strip(
                "'"
            )
        )

    except (
        ValueError,
        IndexError
    ):

        return None


def target_k(
    action
):
    if not action.startswith(
        "set_top_k_"
    ):

        return None

    try:

        return int(
            action.split(
                "_"
            )[-1]
        )

    except ValueError:

        return None


def inspect_topk_policy(
    policy
):
    print("\n" + "=" * 60)
    print("TOP-K POLICY")
    print("=" * 60)

    topk_policy = (
        policy.get(
            "topk_policy",
            {}
        )
    )

    invalid_transitions = []
    threshold_violations = []
    support_warnings = []

    minimum_gain = (
        policy.get(
            "objective",
            {}
        ).get(
            "minimum_gain",
            0.03
        )
    )

    minimum_support = (
        policy.get(
            "objective",
            {}
        ).get(
            "minimum_query_support",
            10
        )
    )

    for state, data in (
        topk_policy.items()
    ):

        action = data.get(
            "selected_action",
            "unknown"
        )

        queries = data.get(
            "query_count",
            0
        )

        candidates = data.get(
            "candidates",
            {}
        )

        keep_utility = candidates.get(
            "keep",
            {
                "average_utility": 0.0
            }
        ).get(
            "average_utility",
            0.0
        )

        print(
            f"\nState: {state}"
        )

        print(
            "  Current K:",
            current_k(state)
        )

        print(
            "  Selected action:",
            action
        )

        print(
            "  Query count:",
            queries
        )

        target = target_k(
            action
        )

        if (
            target is not None
            and
            current_k(state) is not None
            and
            target <= current_k(state)
        ):

            invalid_transitions.append(
                (
                    state,
                    action
                )
            )

        if (
            action != "keep"
            and
            action in candidates
        ):

            selected_utility = (
                candidates[action][
                    "average_utility"
                ]
            )

            gain = (
                selected_utility
                -
                keep_utility
            )

            if gain < minimum_gain:

                threshold_violations.append(
                    (
                        state,
                        action,
                        gain
                    )
                )

        if queries < minimum_support:

            support_warnings.append(
                (
                    state,
                    queries
                )
            )

        print(
            "  Candidate utilities:"
        )

        for candidate_action, candidate in (
            candidates.items()
        ):

            print(
                f"    {candidate_action}: "
                f"{candidate.get('average_utility', 0.0):.6f}"
            )

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    if invalid_transitions:

        print(
            "INVALID TOP-K TRANSITIONS:"
        )

        for state, action in (
            invalid_transitions
        ):

            print(
                f"  {state} -> {action}"
            )

    else:

        print(
            "✓ All selected Top-K "
            "transitions are expansions."
        )

    if threshold_violations:

        print(
            "\nMINIMUM-GAIN VIOLATIONS:"
        )

        for state, action, gain in (
            threshold_violations
        ):

            print(
                f"  {state}: "
                f"{action}, gain={gain:.6f}"
            )

    else:

        print(
            "✓ All selected Top-K "
            "actions satisfy minimum gain."
        )

    if support_warnings:

        print(
            "\nLOW-SUPPORT STATES:"
        )

        for state, queries in (
            support_warnings
        ):

            print(
                f"  {state}: "
                f"{queries} queries"
            )

    else:

        print(
            "✓ All policy states "
            "have sufficient query support."
        )


def main():
    print("=" * 60)
    print(
        "FIQA DEVELOPMENT ACTION "
        "POLICY V5.1 INSPECTION"
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

    print("\n" + "=" * 60)
    print(
        "INSPECTION COMPLETE"
    )
    print("=" * 60)

if __name__ == "__main__":
    main()