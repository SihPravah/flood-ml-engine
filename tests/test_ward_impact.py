import pytest

from pravaha_ml.impact.models import (
    AdministrativeExposure,
    AdministrativeUnitType,
    ImpactLevel,
    RouteReadiness,
    WardHazardInputs,
    WardImpactPolicy,
)
from pravaha_ml.impact.ward import (
    assess_ward_impact,
)


def make_exposure(
    **overrides,
) -> AdministrativeExposure:
    values = {
        "unit_id": "WARD_03",
        "unit_name": "Ward 3",
        "unit_type": (
            AdministrativeUnitType.WARD
        ),
        "population": 4820,
        "road_count": 10,
        "drain_count": 5,
        "critical_facility_count": 2,
        "data_confidence": 0.90,
    }

    values.update(
        overrides
    )

    return AdministrativeExposure(
        **values
    )


def make_hazards(
    **overrides,
) -> WardHazardInputs:
    values = {
        "flood_risk_score": 0.15,
        "flood_confidence": 0.90,
        "landslide_risk_score": 0.15,
        "landslide_confidence": 0.90,
        "cascade_intensity_score": 0.10,
        "cascade_confidence": 0.90,
        "high_risk_road_count": 0,
        "overflowing_drain_count": 0,
        "route_readiness": (
            RouteReadiness.AVAILABLE
        ),
    }

    values.update(
        overrides
    )

    return WardHazardInputs(
        **values
    )


def test_low_hazard_area_has_low_impact():
    result = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(),
    )

    assert (
        result.impact_level
        == ImpactLevel.LOW
    )

    assert result.reliable is True


def test_high_flood_risk_increases_impact():
    low = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            flood_risk_score=0.10
        ),
    )

    high = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            flood_risk_score=0.95
        ),
    )

    assert (
        high.impact_score
        > low.impact_score
    )


def test_landslide_risk_increases_impact():
    low = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            landslide_risk_score=0.10
        ),
    )

    high = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            landslide_risk_score=0.95
        ),
    )

    assert (
        high.impact_score
        > low.impact_score
    )


def test_drain_overflow_increases_infrastructure_pressure():
    none = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            overflowing_drain_count=0
        ),
    )

    overflow = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            overflowing_drain_count=4
        ),
    )

    assert (
        overflow.impact_score
        > none.impact_score
    )


def test_unsafe_route_increases_impact():
    available = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            route_readiness=(
                RouteReadiness.AVAILABLE
            )
        ),
    )

    unsafe = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            route_readiness=(
                RouteReadiness.UNSAFE
            )
        ),
    )

    assert (
        unsafe.impact_score
        > available.impact_score
    )


def test_low_confidence_returns_insufficient_data():
    result = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            flood_confidence=0.40
        ),
    )

    assert result.reliable is False

    assert (
        result.impact_level
        == ImpactLevel.INSUFFICIENT_DATA
    )

    assert (
        "ward_impact_confidence_insufficient"
        in result.reasons
    )


def test_population_is_preserved():
    result = assess_ward_impact(
        exposure=make_exposure(
            population=6500
        ),
        hazards=make_hazards(),
    )

    assert (
        result.population_exposed
        == 6500
    )


def test_unknown_population_is_explicit():
    result = assess_ward_impact(
        exposure=make_exposure(
            population=None
        ),
        hazards=make_hazards(),
    )

    assert (
        result.population_exposed
        is None
    )

    assert (
        "population_exposure_unknown"
        in result.reasons
    )


def test_high_hazard_reasons_are_exposed():
    result = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            flood_risk_score=0.90,
            landslide_risk_score=0.80,
            cascade_intensity_score=0.75,
            high_risk_road_count=3,
            overflowing_drain_count=2,
            route_readiness=(
                RouteReadiness.DEGRADED
            ),
        ),
    )

    assert (
        "high_flash_flood_risk"
        in result.reasons
    )

    assert (
        "high_landslide_susceptibility"
        in result.reasons
    )

    assert (
        "significant_cascade_potential"
        in result.reasons
    )

    assert (
        "high_risk_roads_present"
        in result.reasons
    )

    assert (
        "drainage_overflow_present"
        in result.reasons
    )


def test_unsafe_route_reason_is_exposed():
    result = assess_ward_impact(
        exposure=make_exposure(),
        hazards=make_hazards(
            route_readiness=(
                RouteReadiness.UNSAFE
            )
        ),
    )

    assert (
        "evacuation_route_unsafe"
        in result.reasons
    )


def test_village_unit_supported():
    result = assess_ward_impact(
        exposure=make_exposure(
            unit_id="VILLAGE_01",
            unit_name="Village 1",
            unit_type=(
                AdministrativeUnitType.VILLAGE
            ),
        ),
        hazards=make_hazards(),
    )

    assert (
        result.unit_type
        == AdministrativeUnitType.VILLAGE
    )


def test_invalid_policy_weight_rejected():
    policy = WardImpactPolicy(
        flood_weight=-1.0
    )

    with pytest.raises(
        ValueError,
        match=(
            "Ward impact weights cannot be negative"
        ),
    ):
        assess_ward_impact(
            exposure=make_exposure(),
            hazards=make_hazards(),
            policy=policy,
        )


def test_invalid_threshold_order_rejected():
    policy = WardImpactPolicy(
        watch_threshold=0.60,
        warning_threshold=0.40,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Ward impact thresholds must be ordered"
        ),
    ):
        assess_ward_impact(
            exposure=make_exposure(),
            hazards=make_hazards(),
            policy=policy,
        )