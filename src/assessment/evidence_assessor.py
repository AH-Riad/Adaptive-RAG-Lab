from statistics import mean
from pathlib import Path

from src.core.component import Component
from src.assessment.evidence_result import EvidenceResult
from src.assessment.evidence_features import EvidenceFeatureExtractor
from src.evaluation.evidence_calibrator import EvidenceCalibrator


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

        # This is the OPERATIONAL threshold used by D²RAG.
        #
        # Important:
        # We intentionally DO NOT replace this value
        # with the calibrator's learned threshold.
        #
        # The calibrator still predicts a calibrated probability,
        # but D²RAG uses this stricter threshold to decide
        # whether the retrieved evidence is trustworthy enough
        # to stop the retrieval loop.
        self.acceptance_threshold = (
            acceptance_threshold
        )

        self.feature_extractor = (
            EvidenceFeatureExtractor()
        )

        self.calibrator = None

        # Keep the learned calibration threshold separately
        # for inspection / experiment reporting.
        self.calibrated_threshold = None

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

            # IMPORTANT:
            # Do NOT overwrite self.acceptance_threshold.
            #
            # The calibrator's threshold is stored separately.
            self.calibrated_threshold = (
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

        # CASE 1: NO EVIDENCE

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

        # EXTRACT RETRIEVAL SCORES

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

        # EXTRACT EVIDENCE FEATURES

        features = (
            self.feature_extractor.extract(
                context
            )
        )

        # CALCULATE CONFIDENCE

        if self.calibrator is not None:

            confidence = (
                self.calibrator.predict_probability(
                    features
                )
            )

            decision_mode = (
                "development_calibrated_probability"
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

        # QUERY-SPECIFIC MINIMUM EVIDENCE

        minimum_evidence = (
            self._minimum_evidence_count(
                context
            )
        )

        # ACCEPTANCE DECISION
        
        # D²RAG now requires BOTH:
        #
        # 1. calibrated confidence >= operational threshold
        # 2. enough relevant evidence
        #
        # Notice that the calibrator's learned threshold
        # is NOT used here.
        #
        # Example:
        #
        # calibrated probability = 0.42
        # calibrator threshold   = 0.30
        # operational threshold  = 0.55
        #
        # Result:
        # -> REJECT
        #
        # This is intentional because we want the retrieval
        # controller to have a chance to react to uncertain
        # evidence rather than accepting it immediately.

        confidence_passed = (
            confidence
            >=
            self.acceptance_threshold
        )

        evidence_count_passed = (
            relevant_count
            >=
            minimum_evidence
        )

        accepted = (
            confidence_passed
            and
            evidence_count_passed
        )

        # REASONS / RECOMMENDATIONS

        reasons = []
        recommendations = []

        if accepted:

            reasons.append(
                "Evidence confidence passed "
                f"the operational "
                f"{decision_mode} acceptance "
                "criterion."
            )

            reasons.append(
                "The retrieved evidence also "
                "satisfied the minimum evidence "
                "requirement."
            )

        else:

            reasons.append(
                "Evidence did not satisfy "
                "the D²RAG acceptance criterion."
            )

            if not confidence_passed:

                reasons.append(
                    "Evidence confidence was below "
                    f"the operational threshold "
                    f"({self.acceptance_threshold:.2f})."
                )

                recommendations.append(
                    "Consider changing retrieval "
                    "strategy or increasing Top-K."
                )

            if not evidence_count_passed:

                reasons.append(
                    "The number of relevant evidence "
                    "items was below the minimum "
                    "required for this query type."
                )

                recommendations.append(
                    "Retrieve additional evidence "
                    "appropriate for the query type."
                )

            if coverage < 0.50:

                recommendations.append(
                    "Increase Top-K or change "
                    "retrieval strategy."
                )

        # BUILD RESULT

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