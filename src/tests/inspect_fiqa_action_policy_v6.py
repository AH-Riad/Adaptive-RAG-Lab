import json
from collections import Counter
from pathlib import Path


POLICY_PATH = Path("results/logs/fiqa_dev_action_policy_v6.json")


def summarize_section(name, section):
    actions = Counter()
    supports = []

    for record in section.values():
        action = record.get("selected_action", "keep")
        actions[action] += 1
        candidates = record.get("candidates", {})
        selected = candidates.get(action, {})
        supports.append(int(selected.get("count", 0)))

    print(f"{name} states:", len(section))
    print(f"{name} selected actions:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count}")

    if supports:
        print(
            f"{name} selected support: min={min(supports)} "
            f"mean={sum(supports) / len(supports):.2f} max={max(supports)}"
        )
    print()


def main():
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Policy artifact not found: "
            "results/logs/fiqa_dev_action_policy_v6.json"
        )

    with POLICY_PATH.open("r", encoding="utf-8") as file:
        artifact = json.load(file)

    print("D²RAG FIQA ACTION POLICY V6 INSPECTION")
    print("Version:", artifact.get("version"))
    print("Policy type:", artifact.get("policy_type"))
    print("Training filter:", artifact.get("training_filter"))
    print("Supported Top-K:", artifact.get("supported_top_k"))
    print()

    summarize_section("Strategy policy", artifact.get("strategy_policy", {}))
    summarize_section("Top-K policy", artifact.get("topk_policy", {}))
    summarize_section("Combined policy", artifact.get("combined_policy", {}))

    summary = artifact.get("training_summary", {})
    print("Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("POLICY V6 INSPECTION PASSED")


if __name__ == "__main__":
    main()
