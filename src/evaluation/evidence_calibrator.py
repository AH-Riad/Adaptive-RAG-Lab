import json

import numpy as np

from sklearn.linear_model import LogisticRegression

from src.assessment.evidence_features import (
    EvidenceFeatureExtractor
)


class EvidenceCalibrator:

    def __init__(
        self,
        feature_names=None
    ):

        self.feature_names = (
            feature_names
            if feature_names is not None
            else EvidenceFeatureExtractor.FEATURE_NAMES
        )

        self.model = None
        self.mean = None
        self.scale = None
        self.threshold = 0.5

    def fit(
        self,
        samples,
        labels
    ):

        X = np.array(
            [
                [
                    sample[name]
                    for name in self.feature_names
                ]
                for sample in samples
            ],
            dtype=np.float64
        )

        y = np.array(
            labels,
            dtype=np.int32
        )

        if len(
            np.unique(y)
        ) < 2:

            raise ValueError(
                "Evidence calibration requires "
                "both relevant and irrelevant samples."
            )

        self.mean = X.mean(
            axis=0
        )

        self.scale = X.std(
            axis=0
        )

        self.scale[
            self.scale == 0
        ] = 1.0

        X_scaled = (
            X - self.mean
        ) / self.scale

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        self.model.fit(
            X_scaled,
            y
        )

        probabilities = (
            self.model.predict_proba(
                X_scaled
            )[:, 1]
        )

        self.threshold = (
            self._find_best_threshold(
                probabilities,
                y
            )
        )

        return self

    def predict_probability(
        self,
        features
    ):

        if self.model is None:

            raise RuntimeError(
                "EvidenceCalibrator has not "
                "been fitted."
            )

        X = np.array(
            [
                [
                    features[name]
                    for name in self.feature_names
                ]
            ],
            dtype=np.float64
        )

        X_scaled = (
            X - self.mean
        ) / self.scale

        probability = (
            self.model.predict_proba(
                X_scaled
            )[0, 1]
        )

        return float(
            probability
        )

    def save(
        self,
        path: str,
        dataset: str,
        split: str,
        version: str
    ):

        if self.model is None:

            raise RuntimeError(
                "Cannot save an unfitted calibrator."
            )

        artifact = {
            "dataset":
                dataset,

            "training_split":
                split,

            "version":
                version,

            "feature_names":
                self.feature_names,

            "mean":
                self.mean.tolist(),

            "scale":
                self.scale.tolist(),

            "coefficients":
                self.model.coef_[0].tolist(),

            "intercept":
                float(
                    self.model.intercept_[0]
                ),

            "threshold":
                float(
                    self.threshold
                )
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                artifact,
                file,
                indent=2
            )

    def load(
        self,
        path: str
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            artifact = json.load(
                file
            )

        self.feature_names = (
            artifact["feature_names"]
        )

        self.mean = np.array(
            artifact["mean"],
            dtype=np.float64
        )

        self.scale = np.array(
            artifact["scale"],
            dtype=np.float64
        )

        self.model = LogisticRegression()

        self.model.classes_ = np.array(
            [0, 1]
        )

        self.model.coef_ = np.array(
            [
                artifact[
                    "coefficients"
                ]
            ],
            dtype=np.float64
        )

        self.model.intercept_ = np.array(
            [
                artifact[
                    "intercept"
                ]
            ],
            dtype=np.float64
        )

        self.threshold = float(
            artifact["threshold"]
        )

        return self

    @staticmethod
    def _find_best_threshold(
        probabilities,
        labels
    ):

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.linspace(
            0.05,
            0.95,
            91
        ):

            predictions = (
                probabilities
                >= threshold
            )

            true_positive = np.sum(
                predictions
                &
                (labels == 1)
            )

            false_positive = np.sum(
                predictions
                &
                (labels == 0)
            )

            false_negative = np.sum(
                (~predictions)
                &
                (labels == 1)
            )

            precision = (
                true_positive
                /
                (true_positive + false_positive)
                if (
                    true_positive
                    +
                    false_positive
                )
                else 0.0
            )

            recall = (
                true_positive
                /
                (true_positive + false_negative)
                if (
                    true_positive
                    +
                    false_negative
                )
                else 0.0
            )

            if (
                precision + recall
                == 0
            ):

                f1 = 0.0

            else:

                f1 = (
                    2
                    *
                    precision
                    *
                    recall
                    /
                    (
                        precision
                        +
                        recall
                    )
                )

            if f1 > best_f1:

                best_f1 = f1
                best_threshold = (
                    float(threshold)
                )

        return best_threshold