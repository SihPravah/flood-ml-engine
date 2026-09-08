import pytest

from pravaha_ml.anticipation.forecast import (
    ForecastProvenance,
    ForecastRainfallPoint,
    ForecastScenario,
    ForecastScenarioType,
    RainfallForecast,
)


def test_valid_forecast_point():
    point = ForecastRainfallPoint(
        horizon_minutes=60,
        cumulative_rainfall_mm=40.0,
    )

    assert point.horizon_minutes == 60


def test_invalid_forecast_horizon_rejected():
    with pytest.raises(
        ValueError,
        match="horizon_minutes must be greater than 0",
    ):
        ForecastRainfallPoint(
            horizon_minutes=0,
            cumulative_rainfall_mm=10.0,
        )


def test_negative_forecast_rainfall_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "cumulative_rainfall_mm cannot be negative"
        ),
    ):
        ForecastRainfallPoint(
            horizon_minutes=60,
            cumulative_rainfall_mm=-5.0,
        )


def test_valid_rainfall_forecast():
    forecast = RainfallForecast(
        forecast_id="FC_001",
        points=(
            ForecastRainfallPoint(
                horizon_minutes=30,
                cumulative_rainfall_mm=20.0,
            ),
            ForecastRainfallPoint(
                horizon_minutes=60,
                cumulative_rainfall_mm=40.0,
            ),
        ),
        provenance=(
            ForecastProvenance.NWP_DERIVED
        ),
        forecast_confidence=0.80,
    )

    assert forecast.forecast_id == "FC_001"


def test_duplicate_forecast_horizon_rejected():
    with pytest.raises(
        ValueError,
        match="Forecast horizons must be unique",
    ):
        RainfallForecast(
            forecast_id="FC_001",
            points=(
                ForecastRainfallPoint(
                    horizon_minutes=60,
                    cumulative_rainfall_mm=20.0,
                ),
                ForecastRainfallPoint(
                    horizon_minutes=60,
                    cumulative_rainfall_mm=30.0,
                ),
            ),
            provenance=(
                ForecastProvenance.NWP_DERIVED
            ),
            forecast_confidence=0.80,
        )


def test_non_increasing_horizon_order_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Forecast horizons must be increasing"
        ),
    ):
        RainfallForecast(
            forecast_id="FC_001",
            points=(
                ForecastRainfallPoint(
                    horizon_minutes=60,
                    cumulative_rainfall_mm=40.0,
                ),
                ForecastRainfallPoint(
                    horizon_minutes=30,
                    cumulative_rainfall_mm=20.0,
                ),
            ),
            provenance=(
                ForecastProvenance.NWP_DERIVED
            ),
            forecast_confidence=0.80,
        )


def test_decreasing_cumulative_rainfall_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Cumulative forecast rainfall must be non-decreasing"
        ),
    ):
        RainfallForecast(
            forecast_id="FC_001",
            points=(
                ForecastRainfallPoint(
                    horizon_minutes=30,
                    cumulative_rainfall_mm=30.0,
                ),
                ForecastRainfallPoint(
                    horizon_minutes=60,
                    cumulative_rainfall_mm=20.0,
                ),
            ),
            provenance=(
                ForecastProvenance.NWP_DERIVED
            ),
            forecast_confidence=0.80,
        )


def test_invalid_forecast_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "forecast_confidence must be between 0 and 1"
        ),
    ):
        RainfallForecast(
            forecast_id="FC_001",
            points=(
                ForecastRainfallPoint(
                    horizon_minutes=60,
                    cumulative_rainfall_mm=20.0,
                ),
            ),
            provenance=(
                ForecastProvenance.NWP_DERIVED
            ),
            forecast_confidence=1.20,
        )


def test_invalid_probability_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "probability must be between 0 and 1"
        ),
    ):
        RainfallForecast(
            forecast_id="FC_001",
            points=(
                ForecastRainfallPoint(
                    horizon_minutes=60,
                    cumulative_rainfall_mm=20.0,
                ),
            ),
            provenance=(
                ForecastProvenance.NWP_DERIVED
            ),
            forecast_confidence=0.80,
            probability=1.40,
        )


def test_valid_custom_forecast_scenario():
    scenario = ForecastScenario(
        scenario_id="SC_001",
        scenario_type=(
            ForecastScenarioType.CUSTOM
        ),
        rainfall_points=(
            ForecastRainfallPoint(
                horizon_minutes=60,
                cumulative_rainfall_mm=50.0,
            ),
        ),
        provenance=(
            ForecastProvenance.ESTIMATED
        ),
        forecast_confidence=0.60,
        probability=None,
        rainfall_multiplier=1.0,
    )

    assert scenario.scenario_id == "SC_001"