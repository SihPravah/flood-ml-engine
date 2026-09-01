from pathlib import Path

import pytest

from pravaha_ml.training.pipeline import (
    train_baseline_model,
)
from pravaha_ml.training.serialization import (
    load_model,
    save_model,
)
from pravaha_ml.training.synthetic import (
    generate_synthetic_training_data,
)


def test_save_and_load_model(
    tmp_path: Path,
):
    samples = generate_synthetic_training_data(
        n_samples=100,
        random_state=42,
    )

    result = train_baseline_model(
        samples=samples,
        random_state=42,
    )

    path = (
        tmp_path
        / "baseline_model.pkl"
    )

    save_model(
        model=result.model,
        path=path,
    )

    assert path.exists()

    loaded = load_model(
        path
    )

    assert loaded.is_fitted
    assert (
        loaded.model_version
        == result.model.model_version
    )


def test_missing_model_file_rejected(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "missing.pkl"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_model(path)