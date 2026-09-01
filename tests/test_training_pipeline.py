import pytest

from pravaha_ml.training.pipeline import (
    train_baseline_model,
)
from pravaha_ml.training.synthetic import (
    generate_synthetic_training_data,
)


def test_training_pipeline_runs():
    samples = generate_synthetic_training_data(
        n_samples=200,
        random_state=42,
    )

    result = train_baseline_model(
        samples=samples,
        validation_fraction=0.20,
        random_state=42,
    )

    assert result.model.is_fitted

    assert result.train_size == 160
    assert result.validation_size == 40

    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
    assert 0.0 <= result.metrics.f1 <= 1.0
    assert 0.0 <= result.metrics.roc_auc <= 1.0


def test_training_requires_minimum_samples():
    samples = generate_synthetic_training_data(
        n_samples=8,
        random_state=42,
    )

    with pytest.raises(
        ValueError,
        match="At least 10 samples",
    ):
        train_baseline_model(
            samples=samples
        )


def test_invalid_validation_fraction_rejected():
    samples = generate_synthetic_training_data(
        n_samples=50
    )

    with pytest.raises(
        ValueError,
        match="validation_fraction",
    ):
        train_baseline_model(
            samples=samples,
            validation_fraction=1.5,
        )