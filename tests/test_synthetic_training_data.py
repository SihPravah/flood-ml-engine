import pytest

from pravaha_ml.training.synthetic import (
    generate_synthetic_training_data,
)


def test_generator_returns_requested_sample_count():
    samples = generate_synthetic_training_data(
        n_samples=100,
        random_state=42,
    )

    assert len(samples) == 100


def test_generator_contains_both_classes():
    samples = generate_synthetic_training_data(
        n_samples=100,
        random_state=42,
    )

    labels = {
        sample.label
        for sample in samples
    }

    assert labels == {0, 1}


def test_generator_is_deterministic():
    first = generate_synthetic_training_data(
        n_samples=50,
        random_state=42,
    )

    second = generate_synthetic_training_data(
        n_samples=50,
        random_state=42,
    )

    assert first == second


def test_invalid_sample_count_rejected():
    with pytest.raises(
        ValueError,
        match="n_samples must be at least 2",
    ):
        generate_synthetic_training_data(
            n_samples=1
        )


def test_invalid_flood_fraction_rejected():
    with pytest.raises(
        ValueError,
        match="flood_fraction must be between 0 and 1",
    ):
        generate_synthetic_training_data(
            n_samples=100,
            flood_fraction=1.5,
        )