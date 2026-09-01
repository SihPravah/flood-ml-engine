from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> EvaluationMetrics:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    if not (
        len(y_true)
        == len(y_pred)
        == len(y_score)
    ):
        raise ValueError(
            "y_true, y_pred and y_score "
            "must have equal length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Evaluation data cannot be empty."
        )

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Evaluation data must contain both classes."
        )

    return EvaluationMetrics(
        accuracy=float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        precision=float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        roc_auc=float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
    )