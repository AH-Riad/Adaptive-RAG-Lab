from statistics import mean
from pathlib import Path

from src.core.component import Component
from src.assessment.evidence_result import EvidenceResult
from src.assessment.evidence_features import (
    EvidenceFeatureExtractor
)
from src.evaluation.evidence_calibrator import (
    EvidenceCalibrator
)


class EvidenceAssessor(Component):

    def __init__(
        self,
        relevance_threshold: float = 0.45,
        acceptance_threshold: float = 0.55,
        calibrated_model_path: str | None = None
    ):

        self.relevance_threshold = (
            relevance_threshold
        )

        self.acceptance_threshold = (
            acceptance_threshold
        )

        self.feature_extractor = (
            EvidenceFeatureExtractor()
        )

        self.calibrator = None

        if (
            calibrated_model_path
            and
            Path(
                calibrated_model_path
            ).exists()
        ):

            self.calibrator = (
                EvidenceCalibrator()
            )

            self.calibrator.load(
                calibrated_model_path
            )

            self.acceptance_threshold = (
                self.calibrator.threshold
            )

    def _minimum_evidence_count(
        self,
        context
    ):

        query_analysis = (
            context.query_analysis
        )

        query_type = query_analysis.get(
            "query_type",
            "ambiguous"
        )

        if query_type in {
            "lexical",
            "technical"
        }:
            return 1

        if query_type in {
            "comparison",
            "multi_hop",
            "semantic"
        }:
            return 2

        return 1

    def _calculate_heuristic_confidence(
        self,
        scores
    ):

        top_scores = sorted(
            scores,
            reverse=True
        )[
            :min(3, len(scores))
        ]

        top_evidence_score = mean(
            top_scores
        )

        relevant_count = sum(
            score >= self.relevance_threshold
            for score in scores
        )

        coverage = (
            relevant_count
            /
            len(scores)
        )

        return (
            0.60 * top_evidence_score
            +
            0.40 * coverage
        )

    def run(
        self,
        context
    ):

        retrieval_result = (
            context.retrieval_result
        )

        if retrieval_result is None:

            raise RuntimeError(
                "Evidence assessment requires "
                "retrieval results."
            )

        chunks = (
            retrieval_result.retrieved_chunks
        )

        if not chunks:

            result = EvidenceResult(
                accepted=False,
                confidence=0.0,
                average_score=0.0,
                coverage=0.0,
                retrieved_count=0,
                relevant_count=0,
                threshold=self.acceptance_threshold,
                reasons=[
                    "No evidence was retrieved."
                ],
                recommendations=[
                    "Increase Top-K.",
                    "Change retrieval strategy.",
                    "Rewrite the query."
                ]
            )

            context.evidence_result = result

            context.add_event(
                "evidence_assessment_completed"
            )

            return context

        scores = [
            float(chunk.score)
            for chunk in chunks
        ]

        average_score = mean(
            scores
        )

        relevant_count = sum(
            score >= self.relevance_threshold
            for score in scores
        )

        retrieved_count = len(
            scores
        )

        coverage = (
            relevant_count
            /
            retrieved_count
        )

        features = (
            self.feature_extractor.extract(
                context
            )
        )

        if self.calibrator is not None:

            confidence = (
                self.calibrator.predict_probability(
                    features
                )
            )

            decision_mode = (
                "development_calibrated"
            )

        else:

            confidence = (
                self._calculate_heuristic_confidence(
                    scores
                )
            )

            decision_mode = (
                "heuristic"
            )

        minimum_evidence = (
            self._minimum_evidence_count(
                context
            )
        )

        accepted = (
            confidence
            >=
            self.acceptance_threshold
            and
            relevant_count
            >=
            minimum_evidence
        )

        reasons = []
        recommendations = []

        if accepted:

            reasons.append(
                "Evidence confidence passed "
                f"the frozen {decision_mode} "
                "acceptance criterion."
            )

        else:

            reasons.append(
                "Evidence confidence did not "
                "pass the acceptance criterion."
            )

            if coverage < 0.50:

                recommendations.append(
                    "Increase Top-K or change "
                    "retrieval strategy."
                )

            if relevant_count < minimum_evidence:

                recommendations.append(
                    "Retrieve additional evidence "
                    "appropriate for the query type."
                )

            if confidence < (
                self.acceptance_threshold
            ):

                recommendations.append(
                    "Consider strategy change "
                    "or query rewriting."
                )

        result = EvidenceResult(
            accepted=accepted,
            confidence=confidence,
            average_score=average_score,
            coverage=coverage,
            retrieved_count=retrieved_count,
            relevant_count=relevant_count,
            threshold=self.acceptance_threshold,
            reasons=reasons,
            recommendations=recommendations
        )

        context.evidence_result = result

        context.add_event(
            "evidence_assessment_completed"
        )

        return context