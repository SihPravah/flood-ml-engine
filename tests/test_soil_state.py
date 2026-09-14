import pytest

from pravaha_ml.hydrology.soil_state import (
    MoistureCondition,
    adjust_curve_number,
    calculate_soil_adjusted_runoff,
    classify_soil_moisture,
)


DRY_THRESHOLD = 35.0
WET_THRESHOLD = 70.0


def test_dry_soil_classification():
    condition = classify_soil_moisture(
        soil_moisture_percentage=20.0,
        dry_threshold_percentage=DRY_THRESHOLD,
        wet_threshold_percentage=WET_THRESHOLD,
    )

    assert condition == MoistureCondition.DRY


def test_normal_soil_classification():
    condition = classify_soil_moisture(
        soil_moisture_percentage=50.0,
        dry_threshold_percentage=DRY_THRESHOLD,
        wet_threshold_percentage=WET_THRESHOLD,
    )

    assert condition == MoistureCondition.NORMAL


def test_wet_soil_classification():
    condition = classify_soil_moisture(
        soil_moisture_percentage=82.0,
        dry_threshold_percentage=DRY_THRESHOLD,
        wet_threshold_percentage=WET_THRESHOLD,
    )

    assert condition == MoistureCondition.WET


def test_invalid_soil_moisture_rejected():
    with pytest.raises(
        ValueError,
        match="soil_moisture_percentage must be between 0 and 100",
    ):
        classify_soil_moisture(
            soil_moisture_percentage=120.0,
            dry_threshold_percentage=DRY_THRESHOLD,
            wet_threshold_percentage=WET_THRESHOLD,
        )


def test_invalid_threshold_order_rejected():
    with pytest.raises(
        ValueError,
        match="dry_threshold_percentage must be less than",
    ):
        classify_soil_moisture(
            soil_moisture_percentage=50.0,
            dry_threshold_percentage=80.0,
            wet_threshold_percentage=40.0,
        )


def test_normal_condition_preserves_curve_number():
    adjusted = adjust_curve_number(
        base_curve_number=80.0,
        moisture_condition=MoistureCondition.NORMAL,
    )

    assert adjusted == pytest.approx(80.0)


def test_dry_condition_reduces_curve_number():
    adjusted = adjust_curve_number(
        base_curve_number=80.0,
        moisture_condition=MoistureCondition.DRY,
    )

    assert adjusted < 80.0


def test_wet_condition_increases_curve_number():
    adjusted = adjust_curve_number(
        base_curve_number=80.0,
        moisture_condition=MoistureCondition.WET,
    )

    assert adjusted > 80.0


def test_wet_soil_produces_more_runoff_than_dry_soil():
    dry_result = calculate_soil_adjusted_runoff(
        rainfall_mm=80.0,
        base_curve_number=80.0,
        soil_moisture_percentage=20.0,
        dry_threshold_percentage=DRY_THRESHOLD,
        wet_threshold_percentage=WET_THRESHOLD,
    )

    wet_result = calculate_soil_adjusted_runoff(
        rainfall_mm=80.0,
        base_curve_number=80.0,
        soil_moisture_percentage=85.0,
        dry_threshold_percentage=DRY_THRESHOLD,
        wet_threshold_percentage=WET_THRESHOLD,
    )

    assert (
        wet_result.effective_curve_number
        > dry_result.effective_curve_number
    )

    assert (
        wet_result.runoff.runoff_mm
        > dry_result.runoff.runoff_mm
    )


def test_result_preserves_provenance():
    result = calculate_soil_adjusted_runoff(
        rainfall_mm=80.0,
        base_curve_number=80.0,
        soil_moisture_percentage=82.0,
        dry_threshold_percentage=DRY_THRESHOLD,
        wet_threshold_percentage=WET_THRESHOLD,
    )

    assert result.base_curve_number == 80.0
    assert result.soil_moisture_percentage == 82.0

    assert result.moisture_condition == MoistureCondition.WET

    assert result.effective_curve_number > result.base_curve_number

    assert result.runoff.runoff_mm > 0.0