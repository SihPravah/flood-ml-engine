import numpy as np
import pytest

from pravaha_ml.training.evaluation import (
    evaluate_binary_classifier,
)


def test_perfect_predictions():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 0, 1, 1]
    )

    y_score = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_mismatched_lengths_rejected():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        evaluate_binary_classifier(
            y_true=np.array([0, 1]),
            y_pred=np.array([0]),
            y_score=np.array([0.2, 0.8]),
        )


def test_single_class_rejected():
    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        evaluate_binary_classifier(
            y_true=np.array([0, 0]),
            y_pred=np.array([0, 0]),
            y_score=np.array([0.1, 0.2]),
        )