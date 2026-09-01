import pytest

from pravaha_ml.training.comparison import (
    compare_models,
)
from pravaha_ml.training.realistic_synthetic import (
    generate_realistic_synthetic_data,
)


def test_model_comparison_runs():
    samples = generate_realistic_synthetic_data(
        n_samples=300,
        random_state=42,
    )

    result = compare_models(
        samples=samples,
        validation_fraction=0.20,
        random_state=42,
    )

    assert result.train_size == 240
    assert result.validation_size == 60

    assert (
        result.logistic_regression.model_name
        == "logistic_regression"
    )

    assert (
        result.xgboost.model_name
        == "xgboost"
    )


def test_model_metrics_are_valid():
    samples = generate_realistic_synthetic_data(
        n_samples=300,
        random_state=42,
    )

    result = compare_models(
        samples=samples,
        random_state=42,
    )

    for model_result in [
        result.logistic_regression,
        result.xgboost,
    ]:
        metrics = model_result.metrics

        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0


def test_comparison_requires_minimum_samples():
    samples = generate_realistic_synthetic_data(
        n_samples=10,
        random_state=42,
    )

    with pytest.raises(
        ValueError,
        match="At least 20 samples",
    ):
        compare_models(samples)