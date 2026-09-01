from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from pravaha_ml.models.baseline import (
    BaselineRiskModel,
    hydrology_features_to_vector,
)
from pravaha_ml.training.evaluation import (
    EvaluationMetrics,
    evaluate_binary_classifier,
)
from pravaha_ml.training.synthetic import (
    TrainingSample,
)


@dataclass(frozen=True)
class TrainingResult:
    model: BaselineRiskModel
    metrics: EvaluationMetrics
    train_size: int
    validation_size: int


def train_baseline_model(
    samples: Sequence[TrainingSample],
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> TrainingResult:
    samples = list(samples)

    if len(samples) < 10:
        raise ValueError(
            "At least 10 samples are required."
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
            "Training samples must contain both classes."
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

    train_features = [
        samples[index].features
        for index in train_indices
    ]

    train_labels = [
        samples[index].label
        for index in train_indices
    ]

    validation_samples = [
        samples[index]
        for index in validation_indices
    ]

    model = BaselineRiskModel()

    model.fit(
        feature_rows=train_features,
        labels=train_labels,
    )

    x_validation = np.vstack(
        [
            hydrology_features_to_vector(
                sample.features
            )
            for sample in validation_samples
        ]
    )

    y_validation = np.asarray(
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
        y_true=y_validation,
        y_pred=predictions,
        y_score=probabilities,
    )

    return TrainingResult(
        model=model,
        metrics=metrics,
        train_size=len(train_indices),
        validation_size=len(
            validation_indices
        ),
    )