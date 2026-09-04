from datetime import datetime, timezone

import pytest

from pravaha_ml.data.raw_historical import (
    HistoricalSoilMoistureInput,
    HistoricalStaticCatchmentData,
    SoilMoistureProvenance,
    build_historical_feature_record,
    resolve_soil_moisture,
)
from pravaha_ml.features.rainfall import (
    RainfallObservation,
)


UTC = timezone.utc


def make_rainfall_observations():
    return [
        RainfallObservation(
            timestamp=datetime(
                2026,
                8,
                30,
                12,
                0,
                tzinfo=UTC,
            ),
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026,
                8,
                30,
                12,
                30,
                tzinfo=UTC,
            ),
            rainfall_mm_per_hr=40.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026,
                8,
                30,
                13,
                0,
                tzinfo=UTC,
            ),
            rainfall_mm_per_hr=60.0,
        ),
    ]


def make_catchment():
    return HistoricalStaticCatchmentData(
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.10,
    )


def test_observed_soil_moisture_has_priority():
    resolved = resolve_soil_moisture(
        HistoricalSoilMoistureInput(
            observed_percentage=82.0,
            estimated_percentage=70.0,
        )
    )

    assert (
        resolved.value_percentage
        == pytest.approx(82.0)
    )

    assert (
        resolved.provenance
        == SoilMoistureProvenance.OBSERVED
    )

    assert (
        resolved.was_missing_observation
        is False
    )


def test_estimated_soil_moisture_used_when_observation_missing():
    resolved = resolve_soil_moisture(
        HistoricalSoilMoistureInput(
            observed_percentage=None,
            estimated_percentage=68.0,
        )
    )

    assert (
        resolved.value_percentage
        == pytest.approx(68.0)
    )

    assert (
        resolved.provenance
        == SoilMoistureProvenance.ESTIMATED
    )

    assert (
        resolved.was_missing_observation
        is True
    )


def test_missing_soil_moisture_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Historical soil moisture is unavailable"
        ),
    ):
        resolve_soil_moisture(
            HistoricalSoilMoistureInput()
        )


def test_invalid_observed_soil_moisture_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "observed soil moisture must be between 0 and 100"
        ),
    ):
        resolve_soil_moisture(
            HistoricalSoilMoistureInput(
                observed_percentage=120.0,
            )
        )


def test_invalid_estimated_soil_moisture_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "estimated soil moisture must be between 0 and 100"
        ),
    ):
        resolve_soil_moisture(
            HistoricalSoilMoistureInput(
                estimated_percentage=-5.0,
            )
        )


def test_build_historical_feature_record_with_observed_soil():
    result = (
        build_historical_feature_record(
            event_id="EVENT_001",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=1,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=82.0,
                )
            ),
            catchment=make_catchment(),
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )
    )

    assert (
        result.record.event_id
        == "EVENT_001"
    )

    assert result.record.label == 1

    assert (
        result.training_sample.label
        == 1
    )

    assert (
        result.record.features
        .soil_moisture_percentage
        == pytest.approx(82.0)
    )

    assert (
        result.record.features
        .soil_saturation
        == pytest.approx(0.82)
    )

    assert (
        result.provenance
        .soil_moisture_provenance
        == SoilMoistureProvenance.OBSERVED
    )

    assert (
        result.provenance
        .soil_moisture_was_missing
        is False
    )

    assert (
        result.provenance
        .rainfall_observation_count
        == 3
    )


def test_build_historical_feature_record_with_estimated_soil():
    result = (
        build_historical_feature_record(
            event_id="EVENT_002",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=0,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    estimated_percentage=58.0,
                )
            ),
            catchment=make_catchment(),
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )
    )

    assert (
        result.record.features
        .soil_moisture_percentage
        == pytest.approx(58.0)
    )

    assert (
        result.provenance
        .soil_moisture_provenance
        == SoilMoistureProvenance.ESTIMATED
    )

    assert (
        result.provenance
        .soil_moisture_was_missing
        is True
    )


def test_builder_reuses_hydrology_feature_engine():
    result = (
        build_historical_feature_record(
            event_id="EVENT_003",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=1,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=85.0,
                )
            ),
            catchment=make_catchment(),
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )
    )

    features = result.record.features

    assert features.rain_1h_mm > 0.0
    assert features.api_mm > 0.0

    assert (
        features.effective_curve_number
        > features.base_curve_number
    )

    assert features.runoff_mm >= 0.0

    assert (
        features.concentration_time_minutes
        > 0.0
    )

    assert (
        result.provenance
        .hydrology_feature_engine
        == "pravaha-hydrology-v1"
    )


def test_empty_rainfall_history_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "At least one historical rainfall observation"
        ),
    ):
        build_historical_feature_record(
            event_id="EVENT_004",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=0,
            rainfall_observations=[],
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=50.0,
                )
            ),
            catchment=make_catchment(),
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )


def test_invalid_label_rejected():
    with pytest.raises(
        ValueError,
        match="label must be 0 or 1",
    ):
        build_historical_feature_record(
            event_id="EVENT_005",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=3,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=50.0,
                )
            ),
            catchment=make_catchment(),
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )


def test_invalid_static_curve_number_rejected():
    invalid_catchment = (
        HistoricalStaticCatchmentData(
            base_curve_number=120.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "base_curve_number must be between 1 and 100"
        ),
    ):
        build_historical_feature_record(
            event_id="EVENT_006",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=1,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=80.0,
                )
            ),
            catchment=invalid_catchment,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )


def test_invalid_flow_length_rejected():
    invalid_catchment = (
        HistoricalStaticCatchmentData(
            base_curve_number=80.0,
            flow_length_m=0.0,
            slope_fraction=0.10,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "flow_length_m must be greater than 0"
        ),
    ):
        build_historical_feature_record(
            event_id="EVENT_007",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=1,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=80.0,
                )
            ),
            catchment=invalid_catchment,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )


def test_invalid_slope_rejected():
    invalid_catchment = (
        HistoricalStaticCatchmentData(
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.0,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "slope_fraction must be greater than 0"
        ),
    ):
        build_historical_feature_record(
            event_id="EVENT_008",
            prediction_time=datetime(
                2026,
                8,
                30,
                13,
                30,
                tzinfo=UTC,
            ),
            latitude=30.10,
            longitude=78.20,
            source="historical_test",
            label=1,
            rainfall_observations=(
                make_rainfall_observations()
            ),
            soil_moisture=(
                HistoricalSoilMoistureInput(
                    observed_percentage=80.0,
                )
            ),
            catchment=invalid_catchment,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )