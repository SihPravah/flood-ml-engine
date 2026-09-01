import numpy as np
import pytest

from pravaha_ml.training.thresholds import (
    evaluate_threshold,
    optimize_threshold_for_recall,
)


def test_evaluate_threshold_confusion_matrix():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_score = np.array(
        [0.10, 0.70, 0.80, 0.30]
    )

    result = evaluate_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold=0.50,
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.true_negatives == 1
    assert result.false_negatives == 1

    assert result.precision == pytest.approx(
        0.50
    )

    assert result.recall == pytest.approx(
        0.50
    )

    assert result.specificity == pytest.approx(
        0.50
    )

    assert (
        result.false_positive_rate
        == pytest.approx(0.50)
    )

    assert (
        result.false_negative_rate
        == pytest.approx(0.50)
    )

    assert result.f1 == pytest.approx(
        0.50
    )


def test_lower_threshold_can_increase_recall():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_score = np.array(
        [0.10, 0.20, 0.40, 0.80]
    )

    high_threshold = evaluate_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold=0.50,
    )

    low_threshold = evaluate_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold=0.30,
    )

    assert (
        low_threshold.recall
        >= high_threshold.recall
    )


def test_optimizer_satisfies_minimum_recall():
    y_true = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    y_score = np.array(
        [
            0.10,
            0.20,
            0.40,
            0.35,
            0.60,
            0.80,
        ]
    )

    result = (
        optimize_threshold_for_recall(
            y_true=y_true,
            y_score=y_score,
            minimum_recall=0.66,
        )
    )

    assert (
        result.selected_metrics.recall
        >= 0.66
    )


def test_optimizer_prefers_higher_threshold_when_precision_ties():
    """
    Thresholds 0.40 and 0.50 both achieve precision = 1.0
    and both satisfy the minimum recall requirement.

    The optimizer is designed to prefer the higher threshold
    when precision ties, because the higher threshold is the
    more conservative warning threshold while still satisfying
    the required flood recall.
    """

    y_true = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    y_score = np.array(
        [
            0.10,
            0.20,
            0.30,
            0.40,
            0.70,
            0.90,
        ]
    )

    result = (
        optimize_threshold_for_recall(
            y_true=y_true,
            y_score=y_score,
            minimum_recall=(
                2.0 / 3.0
            ),
            thresholds=[
                0.20,
                0.30,
                0.40,
                0.50,
            ],
        )
    )

    assert (
        result.selected_metrics.recall
        >= 2.0 / 3.0
    )

    assert (
        result.selected_metrics.precision
        == pytest.approx(1.0)
    )

    assert result.selected_threshold == pytest.approx(
        0.50
    )


def test_optimizer_prefers_better_precision():
    """
    When multiple thresholds satisfy the recall target,
    the threshold with higher precision should win.
    """

    y_true = np.array(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    y_score = np.array(
        [
            0.15,
            0.25,
            0.35,
            0.45,
            0.40,
            0.70,
            0.90,
        ]
    )

    low_threshold = evaluate_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold=0.20,
    )

    higher_threshold = evaluate_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold=0.40,
    )

    assert low_threshold.recall == pytest.approx(
        1.0
    )

    assert higher_threshold.recall == pytest.approx(
        1.0
    )

    assert (
        higher_threshold.precision
        > low_threshold.precision
    )

    result = optimize_threshold_for_recall(
        y_true=y_true,
        y_score=y_score,
        minimum_recall=0.90,
        thresholds=[
            0.20,
            0.40,
        ],
    )

    assert result.selected_threshold == pytest.approx(
        0.40
    )


def test_invalid_threshold_rejected():
    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1",
    ):
        evaluate_threshold(
            y_true=[0, 1],
            y_score=[0.20, 0.80],
            threshold=1.20,
        )


def test_invalid_probability_rejected():
    with pytest.raises(
        ValueError,
        match="y_score values must be between 0 and 1",
    ):
        evaluate_threshold(
            y_true=[0, 1],
            y_score=[0.20, 1.50],
            threshold=0.50,
        )


def test_single_class_labels_rejected():
    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        evaluate_threshold(
            y_true=[
                0,
                0,
                0,
            ],
            y_score=[
                0.10,
                0.20,
                0.30,
            ],
            threshold=0.50,
        )


def test_mismatched_lengths_rejected():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        evaluate_threshold(
            y_true=[
                0,
                1,
            ],
            y_score=[
                0.20,
            ],
            threshold=0.50,
        )


def test_invalid_minimum_recall_rejected():
    with pytest.raises(
        ValueError,
        match="minimum_recall must be between 0 and 1",
    ):
        optimize_threshold_for_recall(
            y_true=[
                0,
                1,
            ],
            y_score=[
                0.20,
                0.80,
            ],
            minimum_recall=1.20,
        )


def test_invalid_candidate_threshold_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Candidate thresholds must be between 0 and 1"
        ),
    ):
        optimize_threshold_for_recall(
            y_true=[
                0,
                1,
            ],
            y_score=[
                0.20,
                0.80,
            ],
            minimum_recall=0.50,
            thresholds=[
                0.20,
                1.20,
            ],
        )


def test_empty_candidate_thresholds_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "At least one candidate threshold is required"
        ),
    ):
        optimize_threshold_for_recall(
            y_true=[
                0,
                1,
            ],
            y_score=[
                0.20,
                0.80,
            ],
            minimum_recall=0.50,
            thresholds=[],
        )


def test_impossible_recall_requirement_rejected():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_score = [
        0.10,
        0.20,
        0.30,
        0.40,
    ]

    with pytest.raises(
        ValueError,
        match=(
            "No candidate threshold satisfies"
        ),
    ):
        optimize_threshold_for_recall(
            y_true=y_true,
            y_score=y_score,
            minimum_recall=1.0,
            thresholds=[
                0.90,
                0.95,
            ],
        )