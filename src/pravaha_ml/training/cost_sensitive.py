from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from pravaha_ml.models.baseline import (
    BaselineRiskModel,
    hydrology_features_to_vector,
)
from pravaha_ml.models.xgboost_model import (
    XGBoostRiskModel,
)
from pravaha_ml.training.synthetic import TrainingSample
from pravaha_ml.training.thresholds import (
    ThresholdMetrics,
    optimize_threshold_for_recall,
    evaluate_threshold,
)


@dataclass(frozen=True)
class CostSensitiveModelResult:
    model_name: str
    configuration: str

    selected_threshold: float

    calibration_metrics: ThresholdMetrics
    test_metrics: ThresholdMetrics


@dataclass(frozen=True)
class CostSensitiveComparisonResult:
    logistic_standard: CostSensitiveModelResult
    logistic_balanced: CostSensitiveModelResult

    xgboost_standard: CostSensitiveModelResult
    xgboost_balanced: CostSensitiveModelResult

    train_size: int
    calibration_size: int
    test_size: int


def _scores(
    model,
    samples: list[TrainingSample],
) -> np.ndarray:
    x = np.vstack(
        [
            hydrology_features_to_vector(
                sample.features
            )
            for sample in samples
        ]
    )

    return np.asarray(
        model.model.predict_proba(x)[:, 1],
        dtype=float,
    )


def _labels(
    samples: list[TrainingSample],
) -> np.ndarray:
    return np.asarray(
        [
            sample.label
            for sample in samples
        ],
        dtype=int,
    )


def _fit(
    model,
    samples: list[TrainingSample],
):
    model.fit(
        feature_rows=[
            sample.features
            for sample in samples
        ],
        labels=[
            sample.label
            for sample in samples
        ],
    )

    return model


def _evaluate(
    model,
    model_name: str,
    configuration: str,
    calibration_samples: list[TrainingSample],
    test_samples: list[TrainingSample],
    minimum_recall: float,
) -> CostSensitiveModelResult:
    calibration_labels = _labels(
        calibration_samples
    )

    calibration_scores = _scores(
        model=model,
        samples=calibration_samples,
    )

    threshold_result = (
        optimize_threshold_for_recall(
            y_true=calibration_labels,
            y_score=calibration_scores,
            minimum_recall=minimum_recall,
        )
    )

    threshold = (
        threshold_result.selected_threshold
    )

    test_metrics = evaluate_threshold(
        y_true=_labels(test_samples),
        y_score=_scores(
            model=model,
            samples=test_samples,
        ),
        threshold=threshold,
    )

    return CostSensitiveModelResult(
        model_name=model_name,
        configuration=configuration,
        selected_threshold=threshold,
        calibration_metrics=(
            threshold_result.selected_metrics
        ),
        test_metrics=test_metrics,
    )


def run_cost_sensitive_comparison(
    samples: list[TrainingSample],
    calibration_fraction: float = 0.20,
    test_fraction: float = 0.20,
    minimum_recall: float = 0.85,
    random_state: int = 42,
) -> CostSensitiveComparisonResult:
    samples = list(samples)

    if len(samples) < 100:
        raise ValueError(
            "At least 100 samples are required."
        )

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "calibration_fraction must be between 0 and 1."
        )

    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    if (
        calibration_fraction
        + test_fraction
        >= 1.0
    ):
        raise ValueError(
            "calibration_fraction + test_fraction "
            "must be less than 1."
        )

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError(
            "minimum_recall must be between 0 and 1."
        )

    labels = _labels(samples)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Samples must contain both classes."
        )

    indices = np.arange(
        len(samples)
    )

    train_calibration_indices, test_indices = (
        train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_state,
            stratify=labels,
        )
    )

    remaining_fraction = (
        1.0 - test_fraction
    )

    relative_calibration_fraction = (
        calibration_fraction
        / remaining_fraction
    )

    train_calibration_labels = (
        labels[
            train_calibration_indices
        ]
    )

    train_indices, calibration_indices = (
        train_test_split(
            train_calibration_indices,
            test_size=relative_calibration_fraction,
            random_state=random_state,
            stratify=train_calibration_labels,
        )
    )

    training_samples = [
        samples[index]
        for index in train_indices
    ]

    calibration_samples = [
        samples[index]
        for index in calibration_indices
    ]

    test_samples = [
        samples[index]
        for index in test_indices
    ]

    training_labels = _labels(
        training_samples
    )

    positive_count = int(
        np.sum(
            training_labels == 1
        )
    )

    negative_count = int(
        np.sum(
            training_labels == 0
        )
    )

    if positive_count == 0:
        raise ValueError(
            "Training split contains no positive samples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    logistic_standard_model = _fit(
        BaselineRiskModel(
            random_state=random_state,
            class_weight=None,
        ),
        training_samples,
    )

    logistic_balanced_model = _fit(
        BaselineRiskModel(
            model_version="logreg-balanced-v1",
            random_state=random_state,
            class_weight="balanced",
        ),
        training_samples,
    )

    xgboost_standard_model = _fit(
        XGBoostRiskModel(
            random_state=random_state,
            scale_pos_weight=1.0,
        ),
        training_samples,
    )

    xgboost_balanced_model = _fit(
        XGBoostRiskModel(
            model_version="xgboost-balanced-v1",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        ),
        training_samples,
    )

    return CostSensitiveComparisonResult(
        logistic_standard=_evaluate(
            model=logistic_standard_model,
            model_name="logistic_regression",
            configuration="standard",
            calibration_samples=calibration_samples,
            test_samples=test_samples,
            minimum_recall=minimum_recall,
        ),
        logistic_balanced=_evaluate(
            model=logistic_balanced_model,
            model_name="logistic_regression",
            configuration="balanced",
            calibration_samples=calibration_samples,
            test_samples=test_samples,
            minimum_recall=minimum_recall,
        ),
        xgboost_standard=_evaluate(
            model=xgboost_standard_model,
            model_name="xgboost",
            configuration="standard",
            calibration_samples=calibration_samples,
            test_samples=test_samples,
            minimum_recall=minimum_recall,
        ),
        xgboost_balanced=_evaluate(
            model=xgboost_balanced_model,
            model_name="xgboost",
            configuration="balanced",
            calibration_samples=calibration_samples,
            test_samples=test_samples,
            minimum_recall=minimum_recall,
        ),
        train_size=len(
            training_samples
        ),
        calibration_size=len(
            calibration_samples
        ),
        test_size=len(
            test_samples
        ),
    )