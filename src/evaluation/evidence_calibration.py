import json


class EvidenceCalibrationAnalyzer:

    def __init__(
        self,
        results_path: str,
        number_of_bins: int = 10
    ):

        with open(
            results_path,
            "r",
            encoding="utf-8"
        ) as file:

            all_results = json.load(
                file
            )

        self.results = [
            result
            for result in all_results
            if result.get(
                "system"
            ) == "D2RAG"
        ]

        self.number_of_bins = (
            number_of_bins
        )

    @staticmethod
    def _actual_relevance(
        result
    ) -> int:

        return int(
            result.get(
                "recall_at_5",
                0.0
            ) > 0.0
        )

    def brier_score(self):

        if not self.results:
            return 0.0

        total = 0.0

        for result in self.results:

            confidence = float(
                result.get(
                    "evidence_confidence",
                    0.0
                )
            )

            actual = self._actual_relevance(
                result
            )

            total += (
                confidence - actual
            ) ** 2

        return (
            total
            /
            len(self.results)
        )

    def expected_calibration_error(self):

        if not self.results:
            return 0.0

        bin_data = []

        for bin_index in range(
            self.number_of_bins
        ):

            lower = (
                bin_index
                /
                self.number_of_bins
            )

            upper = (
                (bin_index + 1)
                /
                self.number_of_bins
            )

            values = []

            for result in self.results:

                confidence = float(
                    result.get(
                        "evidence_confidence",
                        0.0
                    )
                )

                if bin_index == (
                    self.number_of_bins - 1
                ):

                    belongs = (
                        lower
                        <= confidence
                        <= upper
                    )

                else:

                    belongs = (
                        lower
                        <= confidence
                        <
                        upper
                    )

                if belongs:

                    values.append(
                        result
                    )

            if not values:
                continue

            average_confidence = (
                sum(
                    float(
                        item.get(
                            "evidence_confidence",
                            0.0
                        )
                    )
                    for item in values
                )
                /
                len(values)
            )

            average_accuracy = (
                sum(
                    self._actual_relevance(
                        item
                    )
                    for item in values
                )
                /
                len(values)
            )

            proportion = (
                len(values)
                /
                len(self.results)
            )

            bin_data.append({
                "lower":
                    lower,

                "upper":
                    upper,

                "count":
                    len(values),

                "average_confidence":
                    average_confidence,

                "actual_relevance_rate":
                    average_accuracy,

                "gap":
                    abs(
                        average_confidence
                        -
                        average_accuracy
                    )
            })

        ece = sum(
            item["gap"] * (
                item["count"]
                /
                len(self.results)
            )
            for item in bin_data
        )

        return ece

    def acceptance_statistics(self):

        accepted = 0
        correctly_accepted = 0
        false_accepted = 0
        actual_relevant = 0
        correctly_retrieved = 0

        for result in self.results:

            predicted = bool(
                result.get(
                    "accepted",
                    False
                )
            )

            actual = (
                self._actual_relevance(
                    result
                )
            )

            if actual:
                actual_relevant += 1

            if predicted:

                accepted += 1

                if actual:

                    correctly_accepted += 1

                else:

                    false_accepted += 1

            if predicted and actual:

                correctly_retrieved += 1

        acceptance_precision = (
            correctly_accepted
            /
            accepted
            if accepted
            else 0.0
        )

        acceptance_recall = (
            correctly_accepted
            /
            actual_relevant
            if actual_relevant
            else 0.0
        )

        false_acceptance_rate = (
            false_accepted
            /
            accepted
            if accepted
            else 0.0
        )

        return {
            "accepted_count":
                accepted,

            "actual_relevant_count":
                actual_relevant,

            "correctly_accepted_count":
                correctly_accepted,

            "false_accepted_count":
                false_accepted,

            "acceptance_precision":
                acceptance_precision,

            "acceptance_recall":
                acceptance_recall,

            "false_acceptance_rate":
                false_acceptance_rate
        }

    def confidence_summary(self):

        if not self.results:

            return {}

        relevant_confidences = []
        irrelevant_confidences = []

        for result in self.results:

            confidence = float(
                result.get(
                    "evidence_confidence",
                    0.0
                )
            )

            actual = (
                self._actual_relevance(
                    result
                )
            )

            if actual:

                relevant_confidences.append(
                    confidence
                )

            else:

                irrelevant_confidences.append(
                    confidence
                )

        result = {}

        if relevant_confidences:

            result[
                "mean_confidence_when_relevant"
            ] = (
                sum(
                    relevant_confidences
                )
                /
                len(
                    relevant_confidences
                )
            )

        else:

            result[
                "mean_confidence_when_relevant"
            ] = 0.0

        if irrelevant_confidences:

            result[
                "mean_confidence_when_irrelevant"
            ] = (
                sum(
                    irrelevant_confidences
                )
                /
                len(
                    irrelevant_confidences
                )
            )

        else:

            result[
                "mean_confidence_when_irrelevant"
            ] = 0.0

        return result

    def analyze(self):

        return {
            "queries":
                len(self.results),

            "brier_score":
                self.brier_score(),

            "expected_calibration_error":
                self.expected_calibration_error(),

            "acceptance":
                self.acceptance_statistics(),

            "confidence":
                self.confidence_summary()
        }