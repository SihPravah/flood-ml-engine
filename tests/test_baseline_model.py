import pytest

from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.models.baseline import (
    BaselineRiskModel,
    FEATURE_NAMES,
    hydrology_features_to_vector,
)


def make_features(
    rain_1h_mm: float,
    soil_saturation: float,
    runoff_mm: float,
    runoff_ratio: float,
) -> HydrologyFeatures:
    return HydrologyFeatures(
        rain_15m_mm=rain_1h_mm / 4.0,
        rain_30m_mm=rain_1h_mm / 2.0,
        rain_1h_mm=rain_1h_mm,
        rain_3h_mm=rain_1h_mm * 1.5,
        rain_6h_mm=rain_1h_mm * 2.0,
        rain_24h_mm=rain_1h_mm * 3.0,
        api_mm=rain_1h_mm * 2.5,

        soil_moisture_percentage=(
            soil_saturation * 100.0
        ),

        soil_saturation=soil_saturation,
        moisture_condition=(
            "WET"
            if soil_saturation >= 0.70
            else "NORMAL"
        ),

        base_curve_number=80.0,

        effective_curve_number=(
            90.0
            if soil_saturation >= 0.70
            else 80.0
        ),

        runoff_mm=runoff_mm,
        runoff_ratio=runoff_ratio,

        flow_length_m=2000.0,
        slope_fraction=0.10,
        concentration_time_minutes=30.0,
    )


def test_feature_vector_has_expected_size():
    features = make_features(
        rain_1h_mm=30.0,
        soil_saturation=0.50,
        runoff_mm=5.0,
        runoff_ratio=0.16,
    )

    vector = hydrology_features_to_vector(
        features
    )

    assert len(vector) == len(FEATURE_NAMES)


def test_model_requires_fit_before_prediction():
    model = BaselineRiskModel()

    features = make_features(
        rain_1h_mm=30.0,
        soil_saturation=0.50,
        runoff_mm=5.0,
        runoff_ratio=0.16,
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted before prediction",
    ):
        model.predict(features)


def test_fit_rejects_mismatched_lengths():
    model = BaselineRiskModel()

    features = [
        make_features(
            rain_1h_mm=20.0,
            soil_saturation=0.40,
            runoff_mm=2.0,
            runoff_ratio=0.10,
        )
    ]

    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        model.fit(
            feature_rows=features,
            labels=[0, 1],
        )


def test_fit_requires_two_classes():
    model = BaselineRiskModel()

    features = [
        make_features(
            rain_1h_mm=20.0,
            soil_saturation=0.40,
            runoff_mm=2.0,
            runoff_ratio=0.10,
        ),
        make_features(
            rain_1h_mm=25.0,
            soil_saturation=0.45,
            runoff_mm=3.0,
            runoff_ratio=0.12,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        model.fit(
            feature_rows=features,
            labels=[0, 0],
        )


def test_baseline_model_can_train_and_predict():
    low_1 = make_features(
        rain_1h_mm=10.0,
        soil_saturation=0.30,
        runoff_mm=1.0,
        runoff_ratio=0.05,
    )

    low_2 = make_features(
        rain_1h_mm=20.0,
        soil_saturation=0.40,
        runoff_mm=3.0,
        runoff_ratio=0.10,
    )

    high_1 = make_features(
        rain_1h_mm=80.0,
        soil_saturation=0.85,
        runoff_mm=45.0,
        runoff_ratio=0.56,
    )

    high_2 = make_features(
        rain_1h_mm=100.0,
        soil_saturation=0.92,
        runoff_mm=65.0,
        runoff_ratio=0.65,
    )

    model = BaselineRiskModel()

    model.fit(
        feature_rows=[
            low_1,
            low_2,
            high_1,
            high_2,
        ],
        labels=[
            0,
            0,
            1,
            1,
        ],
    )

    assert model.is_fitted

    prediction = model.predict(high_2)

    assert 0.0 <= prediction.risk_score <= 1.0
    assert prediction.model_name == "logistic_regression"
    assert prediction.model_version == "logreg-v1"


def test_high_risk_example_scores_above_low_risk_example():
    low = make_features(
        rain_1h_mm=10.0,
        soil_saturation=0.30,
        runoff_mm=1.0,
        runoff_ratio=0.05,
    )

    moderate = make_features(
        rain_1h_mm=30.0,
        soil_saturation=0.50,
        runoff_mm=8.0,
        runoff_ratio=0.20,
    )

    high = make_features(
        rain_1h_mm=100.0,
        soil_saturation=0.92,
        runoff_mm=65.0,
        runoff_ratio=0.65,
    )

    model = BaselineRiskModel()

    model.fit(
        feature_rows=[
            low,
            moderate,
            high,
            make_features(
                rain_1h_mm=90.0,
                soil_saturation=0.88,
                runoff_mm=55.0,
                runoff_ratio=0.61,
            ),
        ],
        labels=[
            0,
            0,
            1,
            1,
        ],
    )

    low_prediction = model.predict(low)
    high_prediction = model.predict(high)

    assert (
        high_prediction.risk_score
        > low_prediction.risk_score
    )