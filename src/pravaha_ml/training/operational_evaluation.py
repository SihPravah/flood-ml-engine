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
    ThresholdOptimizationResult,
    evaluate_threshold,
    optimize_threshold_for_recall,
)


@dataclass(frozen=True)
class OperationalModelEvaluation:
    model_name: str

    calibration_result: ThresholdOptimizationResult

    selected_threshold: float

    test_metrics: ThresholdMetrics


@dataclass(frozen=True)
class OperationalEvaluationResult:
    logistic_regression: OperationalModelEvaluation
    xgboost: OperationalModelEvaluation

    train_size: int
    calibration_size: int
    test_size: int

    minimum_recall: float


def _probabilities(
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


def _fit_model(
    model,
    training_samples: list[TrainingSample],
):
    model.fit(
        feature_rows=[
            sample.features
            for sample in training_samples
        ],
        labels=[
            sample.label
            for sample in training_samples
        ],
    )

    return model


def _evaluate_operational_model(
    model,
    model_name: str,
    calibration_samples: list[TrainingSample],
    test_samples: list[TrainingSample],
    minimum_recall: float,
) -> OperationalModelEvaluation:
    calibration_labels = _labels(
        calibration_samples
    )

    calibration_scores = _probabilities(
        model=model,
        samples=calibration_samples,
    )

    calibration_result = (
        optimize_threshold_for_recall(
            y_true=calibration_labels,
            y_score=calibration_scores,
            minimum_recall=minimum_recall,
        )
    )

    selected_threshold = (
        calibration_result.selected_threshold
    )

    test_labels = _labels(
        test_samples
    )

    test_scores = _probabilities(
        model=model,
        samples=test_samples,
    )

    test_metrics = evaluate_threshold(
        y_true=test_labels,
        y_score=test_scores,
        threshold=selected_threshold,
    )

    return OperationalModelEvaluation(
        model_name=model_name,
        calibration_result=calibration_result,
        selected_threshold=selected_threshold,
        test_metrics=test_metrics,
    )


def run_operational_evaluation(
    samples: list[TrainingSample],
    calibration_fraction: float = 0.20,
    test_fraction: float = 0.20,
    minimum_recall: float = 0.85,
    random_state: int = 42,
) -> OperationalEvaluationResult:
    """
    Perform unbiased development evaluation.

    Data is separated into:

        training set
            -> fit model

        calibration set
            -> choose operating threshold

        test set
            -> evaluate the already-selected threshold

    The test set never participates in threshold selection.

    IMPORTANT:
    Results are still based on development-only synthetic data
    until real historical flash-flood labels are integrated.
    """

    samples = list(samples)

    if len(samples) < 50:
        raise ValueError(
            "At least 50 samples are required."
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

    train_calibration_labels = labels[
        train_calibration_indices
    ]

    remaining_fraction = (
        1.0 - test_fraction
    )

    relative_calibration_fraction = (
        calibration_fraction
        / remaining_fraction
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

    logistic_model = _fit_model(
        model=BaselineRiskModel(
            random_state=random_state,
        ),
        training_samples=training_samples,
    )

    xgboost_model = _fit_model(
        model=XGBoostRiskModel(
            random_state=random_state,
        ),
        training_samples=training_samples,
    )

    logistic_result = (
        _evaluate_operational_model(
            model=logistic_model,
            model_name="logistic_regression",
            calibration_samples=(
                calibration_samples
            ),
            test_samples=test_samples,
            minimum_recall=minimum_recall,
        )
    )

    xgboost_result = (
        _evaluate_operational_model(
            model=xgboost_model,
            model_name="xgboost",
            calibration_samples=(
                calibration_samples
            ),
            test_samples=test_samples,
            minimum_recall=minimum_recall,
        )
    )

    return OperationalEvaluationResult(
        logistic_regression=logistic_result,
        xgboost=xgboost_result,
        train_size=len(training_samples),
        calibration_size=len(
            calibration_samples
        ),
        test_size=len(test_samples),
        minimum_recall=minimum_recall,
    )