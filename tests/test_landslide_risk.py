import pytest

from pravaha_ml.landslide.models import (
    LandslideDataProvenance,
    LandslideInputs,
    LandslideRiskLevel,
    LandslideRiskPolicy,
)
from pravaha_ml.landslide.risk import (
    assess_landslide_susceptibility,
    classify_landslide_risk,
)


def make_inputs(
    **overrides,
) -> LandslideInputs:
    values = {
        "zone_id": "LS_001",
        "slope_degrees": 15.0,
        "soil_saturation": 0.30,
        "rain_1h_mm": 10.0,
        "rain_24h_mm": 30.0,
        "api_mm": 20.0,
        "historical_landslide_score": 0.10,
        "land_cover_disturbance_score": 0.10,
        "data_confidence": 0.95,
        "historical_inventory_provenance": (
            LandslideDataProvenance.VERIFIED
        ),
    }

    values.update(
        overrides
    )

    return LandslideInputs(
        **values
    )


def test_low_hazard_inputs_produce_low_risk():
    result = (
        assess_landslide_susceptibility(
            make_inputs()
        )
    )

    assert (
        result.risk_level
        == LandslideRiskLevel.LOW
    )

    assert result.reliable is True


def test_extreme_conditions_increase_risk():
    low = assess_landslide_susceptibility(
        make_inputs()
    )

    high = assess_landslide_susceptibility(
        make_inputs(
            slope_degrees=45.0,
            soil_saturation=0.95,
            rain_1h_mm=90.0,
            rain_24h_mm=250.0,
            api_mm=180.0,
            historical_landslide_score=0.95,
            land_cover_disturbance_score=0.90,
        )
    )

    assert (
        high.susceptibility_score
        > low.susceptibility_score
    )

    assert (
        high.risk_level
        in {
            LandslideRiskLevel.HIGH,
            LandslideRiskLevel.SEVERE,
        }
    )


def test_steeper_slope_increases_susceptibility():
    gentle = assess_landslide_susceptibility(
        make_inputs(
            slope_degrees=10.0
        )
    )

    steep = assess_landslide_susceptibility(
        make_inputs(
            slope_degrees=40.0
        )
    )

    assert (
        steep.susceptibility_score
        > gentle.susceptibility_score
    )


def test_wetter_soil_increases_susceptibility():
    dry = assess_landslide_susceptibility(
        make_inputs(
            soil_saturation=0.20
        )
    )

    wet = assess_landslide_susceptibility(
        make_inputs(
            soil_saturation=0.95
        )
    )

    assert (
        wet.susceptibility_score
        > dry.susceptibility_score
    )


def test_heavier_recent_rain_increases_susceptibility():
    light = assess_landslide_susceptibility(
        make_inputs(
            rain_1h_mm=5.0
        )
    )

    heavy = assess_landslide_susceptibility(
        make_inputs(
            rain_1h_mm=80.0
        )
    )

    assert (
        heavy.susceptibility_score
        > light.susceptibility_score
    )


def test_historical_inventory_increases_susceptibility():
    weak_history = (
        assess_landslide_susceptibility(
            make_inputs(
                historical_landslide_score=0.0
            )
        )
    )

    strong_history = (
        assess_landslide_susceptibility(
            make_inputs(
                historical_landslide_score=1.0
            )
        )
    )

    assert (
        strong_history.susceptibility_score
        > weak_history.susceptibility_score
    )


def test_estimated_inventory_reduces_confidence():
    verified = (
        assess_landslide_susceptibility(
            make_inputs(
                historical_inventory_provenance=(
                    LandslideDataProvenance.VERIFIED
                )
            )
        )
    )

    estimated = (
        assess_landslide_susceptibility(
            make_inputs(
                historical_inventory_provenance=(
                    LandslideDataProvenance.ESTIMATED
                )
            )
        )
    )

    assert (
        estimated.confidence
        < verified.confidence
    )

    assert (
        "historical_inventory_estimated"
        in estimated.reasons
    )


def test_missing_inventory_reduces_confidence_more():
    estimated = (
        assess_landslide_susceptibility(
            make_inputs(
                historical_inventory_provenance=(
                    LandslideDataProvenance.ESTIMATED
                )
            )
        )
    )

    missing = (
        assess_landslide_susceptibility(
            make_inputs(
                historical_inventory_provenance=(
                    LandslideDataProvenance.MISSING
                )
            )
        )
    )

    assert (
        missing.confidence
        < estimated.confidence
    )


def test_low_confidence_assessment_flagged():
    result = (
        assess_landslide_susceptibility(
            make_inputs(
                data_confidence=0.40,
            )
        )
    )

    assert result.reliable is False

    assert (
        "landslide_assessment_low_confidence"
        in result.reasons
    )


def test_reasons_explain_major_drivers():
    result = (
        assess_landslide_susceptibility(
            make_inputs(
                slope_degrees=40.0,
                soil_saturation=0.90,
                rain_1h_mm=70.0,
                rain_24h_mm=150.0,
                api_mm=120.0,
                historical_landslide_score=0.80,
                land_cover_disturbance_score=0.70,
            )
        )
    )

    assert (
        "steep_terrain"
        in result.reasons
    )

    assert (
        "high_soil_saturation"
        in result.reasons
    )

    assert (
        "intense_recent_rainfall"
        in result.reasons
    )

    assert (
        "high_antecedent_rainfall"
        in result.reasons
    )

    assert (
        "historical_landslide_evidence"
        in result.reasons
    )


def test_risk_classification_boundaries():
    assert (
        classify_landslide_risk(0.10)
        == LandslideRiskLevel.LOW
    )

    assert (
        classify_landslide_risk(0.35)
        == LandslideRiskLevel.WATCH
    )

    assert (
        classify_landslide_risk(0.55)
        == LandslideRiskLevel.WARNING
    )

    assert (
        classify_landslide_risk(0.75)
        == LandslideRiskLevel.HIGH
    )

    assert (
        classify_landslide_risk(0.90)
        == LandslideRiskLevel.SEVERE
    )


def test_invalid_susceptibility_score_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "susceptibility_score must be between 0 and 1"
        ),
    ):
        classify_landslide_risk(
            1.20
        )


def test_invalid_policy_weight_rejected():
    policy = LandslideRiskPolicy(
        slope_weight=-1.0
    )

    with pytest.raises(
        ValueError,
        match=(
            "Landslide susceptibility weights "
            "cannot be negative"
        ),
    ):
        assess_landslide_susceptibility(
            make_inputs(),
            policy=policy,
        )


def test_invalid_threshold_order_rejected():
    policy = LandslideRiskPolicy(
        watch_threshold=0.60,
        warning_threshold=0.40,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Landslide risk thresholds must be ordered"
        ),
    ):
        assess_landslide_susceptibility(
            make_inputs(),
            policy=policy,
        )