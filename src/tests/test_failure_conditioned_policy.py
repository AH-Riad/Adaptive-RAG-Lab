from src.evaluation.action_policy_builder import FailureConditionedActionPolicyBuilder


def make_builder(minimum_gain=0.03, minimum_support=5):
    builder = object.__new__(FailureConditionedActionPolicyBuilder)
    builder.minimum_gain = minimum_gain
    builder.minimum_query_support = minimum_support
    return builder


def test_gain_selects_action():
    builder = make_builder()
    groups = {
        (
            "lexical",
            "dense",
            "low",
            "lexical_strategy_mismatch",
            5,
        ): [
            {"action": "keep", "utility": 0.30, "utility_gain": 0.0},
            {"action": "switch_to_hybrid", "utility": 0.38, "utility_gain": 0.08},
        ] * 6
    }
    policy = builder._aggregate(groups)
    state = str(next(iter(groups)))
    assert policy[state]["selected_action"] == "switch_to_hybrid"


def test_small_gain_keeps():
    builder = make_builder()
    groups = {
        ("ambiguous", "dense", "low", "ranking_uncertainty", 5): [
            {"action": "keep", "utility": 0.30, "utility_gain": 0.0},
            {"action": "set_top_k_10", "utility": 0.32, "utility_gain": 0.02},
        ] * 6
    }
    policy = builder._aggregate(groups)
    state = str(next(iter(groups)))
    assert policy[state]["selected_action"] == "keep"


def test_low_support_action_is_not_selected():
    builder = make_builder(minimum_support=5)
    groups = {
        ("ambiguous", "dense", "low", "ranking_uncertainty", 5): [
            {"action": "keep", "utility": 0.30, "utility_gain": 0.0},
            {"action": "set_top_k_10", "utility": 0.50, "utility_gain": 0.20},
        ]
        + [
            {"action": "keep", "utility": 0.30, "utility_gain": 0.0}
        ] * 5
    }
    policy = builder._aggregate(groups)
    state = str(next(iter(groups)))
    assert policy[state]["selected_action"] == "keep"


def main():
    test_gain_selects_action()
    test_small_gain_keeps()
    test_low_support_action_is_not_selected()
    print("FAILURE-CONDITIONED POLICY TEST PASSED")


if __name__ == "__main__":
    main()
