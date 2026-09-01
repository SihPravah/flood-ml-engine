import pytest

from pravaha_ml.training.operational_evaluation import (
    run_operational_evaluation,
)
from pravaha_ml.training.realistic_synthetic import (
    generate_realistic_synthetic_data,
)


def test_operational_evaluation_runs():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = run_operational_evaluation(
        samples=samples,
        calibration_fraction=0.20,
        test_fraction=0.20,
        minimum_recall=0.80,
        random_state=42,
    )

    assert result.train_size == 600
    assert result.calibration_size == 200
    assert result.test_size == 200


def test_logistic_threshold_selected_from_calibration():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = run_operational_evaluation(
        samples=samples,
        minimum_recall=0.80,
        random_state=42,
    )

    logistic = result.logistic_regression

    assert (
        logistic.selected_threshold
        == logistic.calibration_result.selected_threshold
    )

    assert (
        logistic.calibration_result
        .selected_metrics
        .recall
        >= 0.80
    )


def test_xgboost_threshold_selected_from_calibration():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = run_operational_evaluation(
        samples=samples,
        minimum_recall=0.80,
        random_state=42,
    )

    xgboost = result.xgboost

    assert (
        xgboost.selected_threshold
        == xgboost.calibration_result.selected_threshold
    )

    assert (
        xgboost.calibration_result
        .selected_metrics
        .recall
        >= 0.80
    )


def test_test_metrics_are_valid():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = run_operational_evaluation(
        samples=samples,
        random_state=42,
    )

    for model_result in [
        result.logistic_regression,
        result.xgboost,
    ]:
        metrics = model_result.test_metrics

        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0

        assert (
            0.0
            <= metrics.false_positive_rate
            <= 1.0
        )

        assert (
            0.0
            <= metrics.false_negative_rate
            <= 1.0
        )

        assert 0.0 <= metrics.f1 <= 1.0


def test_small_dataset_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=20,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="At least 50 samples",
    ):
        run_operational_evaluation(
            samples=samples
        )


def test_invalid_calibration_fraction_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=100,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="calibration_fraction",
    ):
        run_operational_evaluation(
            samples=samples,
            calibration_fraction=1.20,
        )


def test_invalid_test_fraction_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=100,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="test_fraction",
    ):
        run_operational_evaluation(
            samples=samples,
            test_fraction=1.20,
        )


def test_combined_holdout_fraction_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=100,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="must be less than 1",
    ):
        run_operational_evaluation(
            samples=samples,
            calibration_fraction=0.50,
            test_fraction=0.50,
        )