from pathlib import Path

from src.adaptation.feedback_controller import FeedbackController
from src.evaluation.action_policy_builder_v7 import DiagnosisFirstActionPolicyBuilder


def test_v7_supported_top_k_is_frozen():
    assert FeedbackController.SUPPORTED_TOP_K == (3, 5, 8, 10, 15)
    assert DiagnosisFirstActionPolicyBuilder.SUPPORTED_TOP_K == (3, 5, 8, 10, 15)


def test_v7_selection_thresholds_are_frozen():
    assert DiagnosisFirstActionPolicyBuilder.MIN_GAIN == 0.02
    assert DiagnosisFirstActionPolicyBuilder.MIN_WIN_RATE == 0.55
    assert DiagnosisFirstActionPolicyBuilder.MAX_HARM_RATE == 0.25
    controller_source = Path("src/adaptation/feedback_controller.py").read_text(encoding="utf-8")
    assert "minimum_improvement: float = 0.02" in controller_source


def test_v7_orchestrator_uses_v7_policy():
    path = Path("src/adaptation/adaptive_retrieval_orchestrator.py")
    text = path.read_text(encoding="utf-8")
    assert "fiqa_dev_action_policy_v7.json" in text
    assert "strict_top_k_integrity" in text


def test_v7_has_no_diagnostic_fallback_source():
    path = Path("src/adaptation/feedback_controller.py")
    text = path.read_text(encoding="utf-8")
    assert "diagnostic_fallback" not in text
