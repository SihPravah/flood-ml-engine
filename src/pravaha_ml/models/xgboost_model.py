from dataclasses import dataclass
from typing import Iterable

import numpy as np
from xgboost import XGBClassifier

from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.models.baseline import (
    hydrology_features_to_vector,
)
from pravaha_ml.models.risk import (
    RiskLevel,
    classify_risk,
)


@dataclass(frozen=True)
class XGBoostRiskPrediction:
    risk_score: float
    risk_level: RiskLevel
    model_name: str
    model_version: str


class XGBoostRiskModel:
    """
    XGBoost flash-flood risk classifier.

    scale_pos_weight controls the relative importance of
    positive flood samples.

    A value greater than 1 increases the cost of missing
    positive-class examples during training.
    """

    def __init__(
        self,
        model_version: str = "xgboost-v1",
        random_state: int = 42,
        scale_pos_weight: float = 1.0,
    ) -> None:
        if scale_pos_weight <= 0.0:
            raise ValueError(
                "scale_pos_weight must be greater than 0."
            )

        self.scale_pos_weight = scale_pos_weight

        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
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
        feature_rows = list(
            feature_rows
        )

        labels = list(
            labels
        )

        if len(feature_rows) == 0:
            raise ValueError(
                "At least one training sample is required."
            )

        if len(feature_rows) != len(labels):
            raise ValueError(
                "feature_rows and labels must have equal length."
            )

        if not set(labels).issubset(
            {0, 1}
        ):
            raise ValueError(
                "labels must contain only 0 or 1."
            )

        if len(set(labels)) < 2:
            raise ValueError(
                "Training data must contain both classes."
            )

        x = np.vstack(
            [
                hydrology_features_to_vector(
                    row
                )
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
    ) -> XGBoostRiskPrediction:
        if not self._is_fitted:
            raise RuntimeError(
                "XGBoostRiskModel must be fitted before prediction."
            )

        vector = (
            hydrology_features_to_vector(
                features
            )
            .reshape(1, -1)
        )

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

        return XGBoostRiskPrediction(
            risk_score=probability,
            risk_level=classify_risk(
                probability
            ),
            model_name="xgboost",
            model_version=self.model_version,
        )