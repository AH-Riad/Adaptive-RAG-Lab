import json
from collections import Counter


POLICY_PATH = "results/logs/fiqa_dev_action_policy_v4.json"


def load_policy():
    with open(POLICY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def format_state(state):
    if isinstance(state, list):
        return tuple(state)
    return state


def inspect_strategy_policy(policy):
    strategy_policy = policy.get("strategy_policy", {})

    print("\n" + "=" * 60)
    print("STRATEGY ACTION POLICY")
    print("=" * 60)

    print(f"Total strategy states: {len(strategy_policy)}")

    action_counts = Counter()

    for state, action_data in strategy_policy.items():
        action = action_data.get("action", "unknown")
        samples = action_data.get("samples", 0)

        action_counts[action] += 1

        print(
            f"State: {state}\n"
            f"  Action: {action}\n"
            f"  Samples: {samples}"
        )

    print("\nStrategy action distribution:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")


def inspect_topk_policy(policy):
    topk_policy = policy.get("topk_policy", {})

    print("\n" + "=" * 60)
    print("TOP-K ACTION POLICY")
    print("=" * 60)

    print(f"Total Top-K states: {len(topk_policy)}")

    action_counts = Counter()

    for state, action_data in topk_policy.items():
        action = action_data.get("action", "unknown")
        samples = action_data.get("samples", 0)

        action_counts[action] += 1

        print(
            f"State: {state}\n"
            f"  Action: {action}\n"
            f"  Samples: {samples}"
        )

    print("\nTop-K action distribution:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")


def inspect_action_quality(policy):
    print("\n" + "=" * 60)
    print("ACTION QUALITY SUMMARY")
    print("=" * 60)

    strategy_policy = policy.get("strategy_policy", {})
    topk_policy = policy.get("topk_policy", {})

    print("\nStrategy actions:")
    for state, data in strategy_policy.items():
        action = data.get("action", "unknown")
        samples = data.get("samples", 0)

        utility = data.get("utility")
        gain = data.get("gain")

        print(
            f"{state}\n"
            f"  action={action}, "
            f"samples={samples}, "
            f"utility={utility}, "
            f"gain={gain}"
        )

    print("\nTop-K actions:")
    for state, data in topk_policy.items():
        action = data.get("action", "unknown")
        samples = data.get("samples", 0)

        utility = data.get("utility")
        gain = data.get("gain")

        print(
            f"{state}\n"
            f"  action={action}, "
            f"samples={samples}, "
            f"utility={utility}, "
            f"gain={gain}"
        )


def inspect_metadata(policy):
    print("\n" + "=" * 60)
    print("POLICY METADATA")
    print("=" * 60)

    objective = policy.get("objective")

    print(f"Objective: {objective}")

    print(f"Minimum gain: {policy.get('minimum_gain')}")
    print(f"Cost weight: {policy.get('cost_weight')}")

    print(f"Strategy states: {len(policy.get('strategy_policy', {}))}")
    print(f"Top-K states: {len(policy.get('topk_policy', {}))}")


def main():
    print("=" * 60)
    print("FIQA DEVELOPMENT ACTION POLICY V4 INSPECTION")
    print("=" * 60)

    policy = load_policy()

    inspect_metadata(policy)
    inspect_strategy_policy(policy)
    inspect_topk_policy(policy)
    inspect_action_quality(policy)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()