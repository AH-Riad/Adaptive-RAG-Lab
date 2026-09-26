import json
import tempfile
from pathlib import Path

from src.core.adaptive_context import AdaptiveContext
from src.assessment.evidence_result import EvidenceResult
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.decision_types import RetrievalDifficulty, RetrievalStrategy
from src.adaptation.feedback_controller import FeedbackController


class StubFeatureExtractor:
    def __init__(self, features):
        self.features = features

    def extract(self, context):
        return dict(self.features)


def make_context(strategy, top_k, confidence, coverage):
    context = AdaptiveContext(query="test query")
    context.query_analysis = {"query_type": "lexical"}
    context.retrieval_plan = RetrievalPlan(
        strategy=RetrievalStrategy(strategy),
        top_k=top_k,
        chunk_size=0,
        chunk_overlap=0,
        difficulty=RetrievalDifficulty.MEDIUM,
    )
    context.evidence_result = EvidenceResult(
        accepted=False,
        confidence=confidence,
        average_score=0.55,
        coverage=coverage,
        retrieved_count=5,
        relevant_count=2,
        threshold=0.55,
    )
    context.feedback_history = []
    return context


def write_policy(path, state, action, gain):
    record = {
        "selected_action": action,
        "candidates": {
            action: {
                "count": 12,
                "average_gain": gain,
                "average_utility": gain,
            },
            "keep": {
                "count": 12,
                "average_gain": 0.0,
                "average_utility": 0.0,
            },
        },
        "samples": 12,
    }
    payload = {
        "version": "v6",
        "strategy_policy": {},
        "topk_policy": {},
        "combined_policy": {},
    }
    payload["strategy_policy"][str(state)] = record
    payload["topk_policy"][str(state)] = record
    payload["combined_policy"][str(state)] = record
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strategy_policy_action():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        state = (
            "lexical",
            "dense",
            "low",
            "lexical_strategy_mismatch",
            5,
        )
        write_policy(path, state, "switch_to_hybrid", 0.08)

        controller = FeedbackController(
            policy_path=str(path),
            minimum_samples=5,
            minimum_improvement=0.03,
        )
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.60,
            "top1_top2_gap": 0.10,
        })

        context = make_context("dense", 5, 0.40, 0.80)
        controller.run(context)
        decision = context.feedback_decision

        assert decision.action == "switch_to_hybrid"
        assert decision.target_strategy == "hybrid"
        assert decision.should_retry is True
        assert decision.source == "failure_conditioned_policy_v6"
        assert decision.expected_improvement >= 0.08


def test_top_k_policy_action():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        state = (
            "lexical",
            "dense",
            "low",
            "ranking_uncertainty",
            5,
        )
        write_policy(path, state, "set_top_k_10", 0.06)

        controller = FeedbackController(
            policy_path=str(path),
            minimum_samples=5,
            minimum_improvement=0.03,
        )
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.60,
            "top1_top2_gap": 0.02,
        })

        context = make_context("dense", 5, 0.40, 0.80)
        controller.run(context)
        decision = context.feedback_decision

        assert decision.action == "set_top_k_10"
        assert decision.target_top_k == 10
        assert decision.should_retry is True
        assert decision.source == "failure_conditioned_policy_v6"
        assert decision.expected_improvement >= 0.06


def test_retry_guard_blocks_repeated_action():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        state = (
            "lexical",
            "dense",
            "low",
            "lexical_strategy_mismatch",
            5,
        )
        write_policy(path, state, "switch_to_hybrid", 0.08)

        controller = FeedbackController(
            policy_path=str(path),
            minimum_samples=5,
            minimum_improvement=0.03,
        )
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.60,
            "top1_top2_gap": 0.10,
        })

        context = make_context("dense", 5, 0.40, 0.80)
        context.feedback_history = [
            {"action": "switch_to_hybrid"}
        ]
        controller.run(context)
        decision = context.feedback_decision

        assert decision.action == "keep"
        assert decision.should_retry is False
        assert decision.source == "retry_guard"


def main():
    test_strategy_policy_action()
    test_top_k_policy_action()
    test_retry_guard_blocks_repeated_action()
    print("FEEDBACK CONTROLLER V6 TEST PASSED")


if __name__ == "__main__":
    main()
