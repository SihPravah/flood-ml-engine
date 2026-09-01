import pytest

from pravaha_ml.training.realistic_synthetic import (
    generate_realistic_synthetic_data,
)


def test_realistic_generator_count():
    samples = generate_realistic_synthetic_data(
        n_samples=200,
        random_state=42,
    )

    assert len(samples) == 200


def test_realistic_generator_has_both_classes():
    samples = generate_realistic_synthetic_data(
        n_samples=200,
        random_state=42,
    )

    labels = {
        sample.label
        for sample in samples
    }

    assert labels == {0, 1}


def test_realistic_generator_is_deterministic():
    first = generate_realistic_synthetic_data(
        n_samples=50,
        random_state=42,
    )

    second = generate_realistic_synthetic_data(
        n_samples=50,
        random_state=42,
    )

    assert first == second


def test_realistic_generator_rejects_small_dataset():
    with pytest.raises(
        ValueError,
        match="n_samples must be at least 10",
    ):
        generate_realistic_synthetic_data(
            n_samples=5
        )