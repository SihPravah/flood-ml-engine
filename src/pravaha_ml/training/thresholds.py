from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    f1: float


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    selected_threshold: float
    selected_metrics: ThresholdMetrics
    candidates_evaluated: int
    minimum_recall: float


def _validate_binary_labels(
    y_true: np.ndarray,
) -> None:
    if len(y_true) == 0:
        raise ValueError(
            "y_true cannot be empty."
        )

    unique_labels = set(
        np.unique(y_true).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "y_true must contain only binary labels 0 and 1."
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "y_true must contain both classes."
        )


def _validate_scores(
    y_score: np.ndarray,
) -> None:
    if len(y_score) == 0:
        raise ValueError(
            "y_score cannot be empty."
        )

    if np.any(
        ~np.isfinite(y_score)
    ):
        raise ValueError(
            "y_score must contain only finite values."
        )

    if np.any(
        (y_score < 0.0)
        | (y_score > 1.0)
    ):
        raise ValueError(
            "y_score values must be between 0 and 1."
        )


def evaluate_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
) -> ThresholdMetrics:
    """
    Evaluate a binary classification threshold.

    For PRAVAHA:

        positive class = flash-flood event
        negative class = non-flood event

    Therefore:

        TP = flood correctly detected
        FN = flood missed
        FP = warning issued for non-flood case
        TN = non-flood case correctly rejected
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_true_array = np.asarray(
        list(y_true),
        dtype=int,
    )

    y_score_array = np.asarray(
        list(y_score),
        dtype=float,
    )

    if len(y_true_array) != len(
        y_score_array
    ):
        raise ValueError(
            "y_true and y_score must have equal length."
        )

    _validate_binary_labels(
        y_true_array
    )

    _validate_scores(
        y_score_array
    )

    y_pred = (
        y_score_array >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision_denominator = (
        tp + fp
    )

    recall_denominator = (
        tp + fn
    )

    specificity_denominator = (
        tn + fp
    )

    precision = (
        float(tp / precision_denominator)
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        float(tp / recall_denominator)
        if recall_denominator > 0
        else 0.0
    )

    specificity = (
        float(
            tn
            / specificity_denominator
        )
        if specificity_denominator > 0
        else 0.0
    )

    false_positive_rate = (
        float(
            fp
            / specificity_denominator
        )
        if specificity_denominator > 0
        else 0.0
    )

    false_negative_rate = (
        float(
            fn
            / recall_denominator
        )
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = (
        precision + recall
    )

    f1 = (
        float(
            2.0
            * precision
            * recall
            / f1_denominator
        )
        if f1_denominator > 0
        else 0.0
    )

    return ThresholdMetrics(
        threshold=float(
            threshold
        ),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        precision=precision,
        recall=recall,
        specificity=specificity,
        false_positive_rate=(
            false_positive_rate
        ),
        false_negative_rate=(
            false_negative_rate
        ),
        f1=f1,
    )


def optimize_threshold_for_recall(
    y_true: Iterable[int],
    y_score: Iterable[float],
    minimum_recall: float = 0.85,
    thresholds: Iterable[float] | None = None,
) -> ThresholdOptimizationResult:
    """
    Select an operational threshold for a flood-warning model.

    Strategy:

    1. Require recall >= minimum_recall.
    2. Among thresholds satisfying the recall requirement,
       choose the one with the highest precision.
    3. If precision ties, prefer the higher threshold because
       it generally produces fewer false alarms.

    This is a development strategy.

    Final operational thresholds must be calibrated using
    real historical flood-event data and government warning
    requirements.
    """

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError(
            "minimum_recall must be between 0 and 1."
        )

    y_true_array = np.asarray(
        list(y_true),
        dtype=int,
    )

    y_score_array = np.asarray(
        list(y_score),
        dtype=float,
    )

    if len(y_true_array) != len(
        y_score_array
    ):
        raise ValueError(
            "y_true and y_score must have equal length."
        )

    _validate_binary_labels(
        y_true_array
    )

    _validate_scores(
        y_score_array
    )

    if thresholds is None:
        candidate_thresholds = (
            np.linspace(
                0.05,
                0.95,
                91,
            )
        )
    else:
        candidate_thresholds = np.asarray(
            list(thresholds),
            dtype=float,
        )

    if len(candidate_thresholds) == 0:
        raise ValueError(
            "At least one candidate threshold is required."
        )

    if np.any(
        (candidate_thresholds < 0.0)
        | (candidate_thresholds > 1.0)
    ):
        raise ValueError(
            "Candidate thresholds must be between 0 and 1."
        )

    evaluated = [
        evaluate_threshold(
            y_true=y_true_array,
            y_score=y_score_array,
            threshold=float(
                threshold
            ),
        )
        for threshold in candidate_thresholds
    ]

    eligible = [
        result
        for result in evaluated
        if result.recall
        >= minimum_recall
    ]

    if not eligible:
        raise ValueError(
            "No candidate threshold satisfies "
            "the minimum recall requirement."
        )

    selected = max(
        eligible,
        key=lambda result: (
            result.precision,
            result.threshold,
        ),
    )

    return ThresholdOptimizationResult(
        selected_threshold=(
            selected.threshold
        ),
        selected_metrics=selected,
        candidates_evaluated=len(
            evaluated
        ),
        minimum_recall=(
            minimum_recall
        ),
    )