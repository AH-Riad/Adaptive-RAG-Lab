from src.evaluation.action_policy_builder_v7 import DiagnosisFirstActionPolicyBuilder


def test_aggregate_selects_supported_positive_action():
    builder = DiagnosisFirstActionPolicyBuilder.__new__(DiagnosisFirstActionPolicyBuilder)
    groups = {
        ("lexical", "dense", "low", "lexical_strategy_mismatch", 5): [
            {"action": "keep", "utility": 0.40, "utility_gain": 0.0}
            for _ in range(10)
        ] + [
            {"action": "switch_to_hybrid", "utility": 0.48, "utility_gain": 0.08}
            for _ in range(10)
        ],
    }

    artifact = builder._aggregate(groups, min_support=8)
    record = next(iter(artifact.values()))
    assert record["selected_action"] == "switch_to_hybrid"
    assert record["policy_eligible"] is True


def test_aggregate_rejects_low_win_rate_action():
    builder = DiagnosisFirstActionPolicyBuilder.__new__(DiagnosisFirstActionPolicyBuilder)
    groups = {
        ("lexical", "dense", "low", "lexical_strategy_mismatch", 5): [
            {"action": "keep", "utility": 0.40, "utility_gain": 0.0}
            for _ in range(10)
        ] + [
            {"action": "switch_to_hybrid", "utility": 0.48, "utility_gain": 0.08}
            for _ in range(4)
        ] + [
            {"action": "switch_to_hybrid", "utility": 0.30, "utility_gain": -0.10}
            for _ in range(6)
        ],
    }

    artifact = builder._aggregate(groups, min_support=8)
    record = next(iter(artifact.values()))
    assert record["selected_action"] == "keep"
    assert record["policy_eligible"] is False


def main():
    test_aggregate_selects_supported_positive_action()
    test_aggregate_rejects_low_win_rate_action()
    print("FAILURE-CONDITIONED POLICY V7 TEST PASSED")


if __name__ == "__main__":
    main()
