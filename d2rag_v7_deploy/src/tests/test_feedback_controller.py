import json
import tempfile
from pathlib import Path

from src.adaptation.feedback_controller import FeedbackController
from src.assessment.evidence_result import EvidenceResult
from src.core.adaptive_context import AdaptiveContext
from src.planning.decision_types import RetrievalDifficulty, RetrievalStrategy
from src.planning.retrieval_plan import RetrievalPlan


class StubFeatureExtractor:
    def __init__(self, features):
        self.features = features

    def extract(self, context):
        return dict(self.features)


def make_context(strategy="dense", top_k=5, confidence=0.40, query_type="lexical"):
    context = AdaptiveContext(query="test query")
    context.query_analysis = {"query_type": query_type}
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
        coverage=0.80,
        retrieved_count=5,
        relevant_count=2,
        threshold=0.55,
    )
    context.feedback_history = []
    return context


def policy_record(action, gain, count=12):
    return {
        "selected_action": action,
        "candidates": {
            action: {
                "count": count,
                "average_gain": gain,
                "average_utility": gain,
                "win_rate": 0.75,
                "harm_rate": 0.10,
            },
            "keep": {
                "count": count,
                "average_gain": 0.0,
                "average_utility": 0.0,
                "win_rate": 0.0,
                "harm_rate": 0.0,
            },
        },
        "samples": count,
        "policy_eligible": action != "keep",
    }


def write_policy(path, exact_state=None, exact_action=None, backoff_state=None, backoff_action=None):
    payload = {
        "version": "v7",
        "strategy_policy": {},
        "topk_policy": {},
        "combined_policy": {},
        "strategy_backoff_query_type": {},
        "strategy_backoff_strategy": {},
        "strategy_backoff_diagnosis": {},
        "topk_backoff_query_type": {},
        "topk_backoff_strategy": {},
        "topk_backoff_diagnosis": {},
        "combined_backoff_query_type": {},
        "combined_backoff_strategy": {},
        "combined_backoff_diagnosis": {},
    }

    if exact_state is not None:
        payload["strategy_policy"][str(exact_state)] = policy_record(exact_action, 0.08)

    if backoff_state is not None:
        payload["strategy_backoff_query_type"][str(backoff_state)] = policy_record(backoff_action, 0.07, count=24)

    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strategy_policy_action():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        state = ("lexical", "dense", "low", "lexical_strategy_mismatch", 5)
        write_policy(path, exact_state=state, exact_action="switch_to_hybrid")

        controller = FeedbackController(str(path))
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.60,
            "top1_top2_gap": 0.10,
        })
        context = make_context()
        controller.run(context)
        decision = context.feedback_decision

        assert decision.action == "switch_to_hybrid"
        assert decision.target_strategy == "hybrid"
        assert decision.should_retry is True
        assert decision.source == "failure_conditioned_policy_v7_exact"


def test_hierarchical_backoff_action():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        backoff_state = ("lexical", "dense", "lexical_strategy_mismatch", 5)
        write_policy(path, backoff_state=backoff_state, backoff_action="switch_to_hybrid")

        controller = FeedbackController(str(path))
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.60,
            "top1_top2_gap": 0.10,
        })
        context = make_context()
        controller.run(context)
        decision = context.feedback_decision

        assert decision.action == "switch_to_hybrid"
        assert decision.source == "failure_conditioned_policy_v7_query_type"
        assert decision.metadata["policy_level"] == "query_type"


def test_diagnosis_priority_exposes_strategy_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        write_policy(path)

        controller = FeedbackController(str(path))
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.80,
            "top1_top2_gap": 0.01,
        })
        context = make_context(query_type="ambiguous")
        controller.run(context)
        assert context.feedback_decision.diagnosis == "ambiguous_strategy_risk"


def test_no_heuristic_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        write_policy(path)

        controller = FeedbackController(str(path))
        controller.feature_extractor = StubFeatureExtractor({
            "dense_bm25_agreement": None,
            "top1_score": 0.30,
            "top1_top2_gap": 0.20,
        })
        context = make_context(query_type="ambiguous")
        controller.run(context)
        assert context.feedback_decision.action == "keep"
        assert context.feedback_decision.source == "policy_no_supported_action"


def test_retry_guard_blocks_repeated_action():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        state = ("lexical", "dense", "low", "lexical_strategy_mismatch", 5)
        write_policy(path, exact_state=state, exact_action="switch_to_hybrid")

        controller = FeedbackController(str(path))
        context = make_context()
        context.feedback_history = [{"action": "switch_to_hybrid"}]
        controller.run(context)

        assert context.feedback_decision.action == "keep"
        assert context.feedback_decision.source == "retry_guard"
        assert context.feedback_decision.should_retry is False


def main():
    test_strategy_policy_action()
    test_hierarchical_backoff_action()
    test_diagnosis_priority_exposes_strategy_mismatch()
    test_no_heuristic_fallback()
    test_retry_guard_blocks_repeated_action()
    print("FEEDBACK CONTROLLER V7 TEST PASSED")


if __name__ == "__main__":
    main()
