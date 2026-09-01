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
from pravaha_ml.training.evaluation import (
    EvaluationMetrics,
    evaluate_binary_classifier,
)
from pravaha_ml.training.synthetic import (
    TrainingSample,
)


@dataclass(frozen=True)
class ModelEvaluationResult:
    model_name: str
    metrics: EvaluationMetrics


@dataclass(frozen=True)
class ModelComparisonResult:
    logistic_regression: ModelEvaluationResult
    xgboost: ModelEvaluationResult
    train_size: int
    validation_size: int


def _evaluate_model(
    model,
    model_name: str,
    validation_samples: list[TrainingSample],
) -> ModelEvaluationResult:
    x_validation = np.vstack(
        [
            hydrology_features_to_vector(
                sample.features
            )
            for sample in validation_samples
        ]
    )

    y_true = np.asarray(
        [
            sample.label
            for sample in validation_samples
        ],
        dtype=int,
    )

    probabilities = (
        model.model.predict_proba(
            x_validation
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_pred=predictions,
        y_score=probabilities,
    )

    return ModelEvaluationResult(
        model_name=model_name,
        metrics=metrics,
    )


def compare_models(
    samples: list[TrainingSample],
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> ModelComparisonResult:
    if len(samples) < 20:
        raise ValueError(
            "At least 20 samples are required for comparison."
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    labels = np.asarray(
        [
            sample.label
            for sample in samples
        ],
        dtype=int,
    )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Samples must contain both classes."
        )

    indices = np.arange(
        len(samples)
    )

    train_indices, validation_indices = (
        train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=labels,
        )
    )

    training_samples = [
        samples[index]
        for index in train_indices
    ]

    validation_samples = [
        samples[index]
        for index in validation_indices
    ]

    training_features = [
        sample.features
        for sample in training_samples
    ]

    training_labels = [
        sample.label
        for sample in training_samples
    ]

    logistic_model = (
        BaselineRiskModel()
    )

    logistic_model.fit(
        feature_rows=training_features,
        labels=training_labels,
    )

    xgboost_model = (
        XGBoostRiskModel(
            random_state=random_state
        )
    )

    xgboost_model.fit(
        feature_rows=training_features,
        labels=training_labels,
    )

    logistic_result = _evaluate_model(
        model=logistic_model,
        model_name="logistic_regression",
        validation_samples=validation_samples,
    )

    xgboost_result = _evaluate_model(
        model=xgboost_model,
        model_name="xgboost",
        validation_samples=validation_samples,
    )

    return ModelComparisonResult(
        logistic_regression=logistic_result,
        xgboost=xgboost_result,
        train_size=len(
            train_indices
        ),
        validation_size=len(
            validation_indices
        ),
    )