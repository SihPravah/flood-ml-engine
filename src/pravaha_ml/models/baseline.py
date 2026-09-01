from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.models.risk import (
    RiskLevel,
    classify_risk,
)


FEATURE_NAMES = [
    "rain_15m_mm",
    "rain_30m_mm",
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "api_mm",
    "soil_saturation",
    "base_curve_number",
    "effective_curve_number",
    "runoff_mm",
    "runoff_ratio",
    "flow_length_m",
    "slope_fraction",
    "concentration_time_minutes",
]


@dataclass(frozen=True)
class RiskPrediction:
    risk_score: float
    risk_level: RiskLevel
    model_name: str
    model_version: str


def hydrology_features_to_vector(
    features: HydrologyFeatures,
) -> np.ndarray:
    """
    Convert HydrologyFeatures into PRAVAHA's stable
    numeric model feature order.
    """

    values = asdict(features)

    return np.array(
        [
            float(values[name])
            for name in FEATURE_NAMES
        ],
        dtype=float,
    )


class BaselineRiskModel:
    """
    Logistic Regression baseline for PRAVAHA.

    StandardScaler is included inside the pipeline because
    hydrological variables have very different numeric scales.

    class_weight can optionally be used for cost-sensitive
    learning.

    Examples:

        class_weight=None
            standard Logistic Regression

        class_weight="balanced"
            automatically increases the relative importance
            of the minority class

    The positive class represents a flash-flood event.
    """

    def __init__(
        self,
        model_version: str = "logreg-v1",
        random_state: int = 42,
        class_weight: str | dict | None = None,
    ) -> None:
        self.class_weight = class_weight

        self.model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )

        self.model_version = model_version
        self.random_state = random_state

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        feature_rows: Iterable[HydrologyFeatures],
        labels: Iterable[int],
    ) -> None:
        feature_rows = list(feature_rows)
        labels = list(labels)

        if len(feature_rows) == 0:
            raise ValueError(
                "At least one training sample is required."
            )

        if len(feature_rows) != len(labels):
            raise ValueError(
                "feature_rows and labels must have equal length."
            )

        if not set(labels).issubset({0, 1}):
            raise ValueError(
                "labels must contain only 0 or 1."
            )

        if len(set(labels)) < 2:
            raise ValueError(
                "Training data must contain both classes."
            )

        x = np.vstack(
            [
                hydrology_features_to_vector(row)
                for row in feature_rows
            ]
        )

        y = np.asarray(
            labels,
            dtype=int,
        )

        self.model.fit(
            x,
            y,
        )

        self._is_fitted = True

    def predict(
        self,
        features: HydrologyFeatures,
    ) -> RiskPrediction:
        if not self._is_fitted:
            raise RuntimeError(
                "BaselineRiskModel must be fitted before prediction."
            )

        vector = hydrology_features_to_vector(
            features
        ).reshape(1, -1)

        probability = float(
            self.model.predict_proba(
                vector
            )[0, 1]
        )

        probability = max(
            0.0,
            min(
                probability,
                1.0,
            ),
        )

        return RiskPrediction(
            risk_score=probability,
            risk_level=classify_risk(
                probability
            ),
            model_name="logistic_regression",
            model_version=self.model_version,
        )