from statistics import mean

from src.core.component import Component
from src.assessment.evidence_result import EvidenceResult


class EvidenceAssessor(Component):
    """
    Evaluates retrieved chunks using normalized relevance scores.

    Higher score means stronger evidence.
    """

    def __init__(
        self,
        relevance_threshold: float = 0.60,
        acceptance_threshold: float = 0.65,
    ):
        self.relevance_threshold = relevance_threshold
        self.acceptance_threshold = acceptance_threshold

    def run(self, context):

        retrieval_result = context.retrieval_result

        if retrieval_result is None:
            raise RuntimeError(
                "Evidence assessment requires retrieval results."
            )

        chunks = retrieval_result.retrieved_chunks

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
                    "No chunks were retrieved."
                ],
                recommendations=[
                    "Increase Top-K.",
                    "Change retrieval strategy.",
                    "Rewrite the query.",
                ],
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

        average_score = mean(scores)

        relevant_count = sum(
            score >= self.relevance_threshold
            for score in scores
        )

        retrieved_count = len(scores)

        coverage = (
            relevant_count / retrieved_count
        )

        confidence = (
            0.5 * average_score
            +
            0.5 * coverage
        )

        accepted = (
            confidence >= self.acceptance_threshold
        )

        reasons = []

        recommendations = []

        if accepted:

            reasons.append(
                "Retrieved evidence passed the acceptance threshold."
            )

        else:

            reasons.append(
                "Retrieved evidence is below the acceptance threshold."
            )

            if coverage < 0.50:

                recommendations.append(
                    "Increase Top-K or change retrieval strategy."
                )

            if average_score < self.relevance_threshold:

                recommendations.append(
                    "Consider query rewriting or reranking."
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
            recommendations=recommendations,
        )

        context.evidence_result = result

        context.add_event(
            "evidence_assessment_completed"
        )

        return context