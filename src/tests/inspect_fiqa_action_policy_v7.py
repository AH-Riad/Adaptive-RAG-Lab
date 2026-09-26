import json
from collections import Counter


PATH = "results/logs/fiqa_dev_action_policy_v7.json"


def inspect_section(name, section):
    actions = Counter()
    supported = []
    eligible = 0

    for record in section.values():
        action = record.get("selected_action", "keep")
        actions[action] += 1
        selected = record.get("selected_record", {})
        supported.append(int(record.get("samples", 0)))
        if action != "keep" and record.get("policy_eligible"):
            eligible += 1

    print()
    print(name)
    print("  states:", len(section))
    print("  selected actions:")
    for action, count in sorted(actions.items()):
        print(f"    {action}: {count}")
    if supported:
        print(
            "  support:",
            "min=",
            min(supported),
            "mean=",
            round(sum(supported) / len(supported), 2),
            "max=",
            max(supported),
        )
    print("  eligible non-KEEP states:", eligible)


def main():
    print("D²RAG FIQA ACTION POLICY V7 INSPECTION")
    with open(PATH, "r", encoding="utf-8") as file:
        artifact = json.load(file)

    print("Version:", artifact["version"])
    print("Policy type:", artifact["policy_type"])
    print("Supported Top-K:", artifact["supported_top_k"])
    print("Diagnosis priority:", ", ".join(artifact["diagnosis_priority"]))

    inspect_section("Strategy exact policy", artifact["strategy_policy"])
    inspect_section("Top-K exact policy", artifact["topk_policy"])
    inspect_section("Combined exact policy", artifact["combined_policy"])
    inspect_section("Strategy query-type backoff", artifact["strategy_backoff_query_type"])
    inspect_section("Strategy strategy-level backoff", artifact["strategy_backoff_strategy"])
    inspect_section("Strategy diagnosis-level backoff", artifact["strategy_backoff_diagnosis"])

    print()
    print("Training summary:")
    for key, value in artifact["training_summary"].items():
        print(f"  {key}: {value}")

    assert artifact["version"] == "v7"
    assert artifact["state_definition"] == [
        "query_type",
        "current_strategy",
        "confidence_bucket",
        "diagnosis",
        "current_top_k",
    ]
    print()
    print("POLICY V7 INSPECTION PASSED")


if __name__ == "__main__":
    main()
