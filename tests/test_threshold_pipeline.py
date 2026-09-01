import pytest

from pravaha_ml.training.realistic_synthetic import (
    generate_realistic_synthetic_data,
)
from pravaha_ml.training.threshold_pipeline import (
    run_threshold_pipeline,
)


def test_threshold_pipeline_runs():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=500,
            random_state=42,
        )
    )

    result = run_threshold_pipeline(
        samples=samples,
        validation_fraction=0.20,
        minimum_recall=0.80,
        random_state=42,
    )

    assert result.train_size == 400
    assert result.validation_size == 100

    assert (
        result.logistic_regression.model_name
        == "logistic_regression"
    )

    assert (
        result.xgboost.model_name
        == "xgboost"
    )


def test_optimized_logistic_threshold_meets_recall_target():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = run_threshold_pipeline(
        samples=samples,
        minimum_recall=0.80,
        random_state=42,
    )

    metrics = (
        result.logistic_regression
        .optimized_threshold
        .selected_metrics
    )

    assert metrics.recall >= 0.80


def test_optimized_xgboost_threshold_meets_recall_target():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = run_threshold_pipeline(
        samples=samples,
        minimum_recall=0.80,
        random_state=42,
    )

    metrics = (
        result.xgboost
        .optimized_threshold
        .selected_metrics
    )

    assert metrics.recall >= 0.80


def test_default_threshold_is_preserved_for_comparison():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=500,
            random_state=42,
        )
    )

    result = run_threshold_pipeline(
        samples=samples,
        minimum_recall=0.80,
        random_state=42,
    )

    assert (
        result.logistic_regression
        .default_threshold_metrics
        .threshold
        == 0.50
    )

    assert (
        result.xgboost
        .default_threshold_metrics
        .threshold
        == 0.50
    )


def test_invalid_validation_fraction_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=100,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="validation_fraction",
    ):
        run_threshold_pipeline(
            samples=samples,
            validation_fraction=1.20,
        )


def test_invalid_minimum_recall_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=100,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="minimum_recall",
    ):
        run_threshold_pipeline(
            samples=samples,
            minimum_recall=1.20,
        )