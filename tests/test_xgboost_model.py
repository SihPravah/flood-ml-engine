import pytest

from pravaha_ml.models.xgboost_model import (
    XGBoostRiskModel,
)
from pravaha_ml.training.realistic_synthetic import (
    generate_realistic_synthetic_data,
)


def test_xgboost_requires_fit():
    samples = generate_realistic_synthetic_data(
        n_samples=50,
        random_state=42,
    )

    model = XGBoostRiskModel()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before prediction",
    ):
        model.predict(
            samples[0].features
        )


def test_xgboost_can_train():
    samples = generate_realistic_synthetic_data(
        n_samples=100,
        random_state=42,
    )

    model = XGBoostRiskModel()

    model.fit(
        feature_rows=[
            sample.features
            for sample in samples
        ],
        labels=[
            sample.label
            for sample in samples
        ],
    )

    assert model.is_fitted


def test_xgboost_prediction_valid():
    samples = generate_realistic_synthetic_data(
        n_samples=100,
        random_state=42,
    )

    model = XGBoostRiskModel()

    model.fit(
        feature_rows=[
            sample.features
            for sample in samples
        ],
        labels=[
            sample.label
            for sample in samples
        ],
    )

    prediction = model.predict(
        samples[0].features
    )

    assert (
        0.0
        <= prediction.risk_score
        <= 1.0
    )

    assert (
        prediction.model_name
        == "xgboost"
    )