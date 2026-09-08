import pytest

from pravaha_ml.impact.models import (
    AdministrativeExposure,
    AdministrativeUnitType,
    RouteReadiness,
    WardHazardInputs,
)


def test_valid_administrative_exposure():
    exposure = AdministrativeExposure(
        unit_id="WARD_03",
        unit_name="Ward 3",
        unit_type=(
            AdministrativeUnitType.WARD
        ),
        population=4820,
        road_count=15,
        drain_count=6,
        critical_facility_count=2,
        data_confidence=0.90,
    )

    assert exposure.population == 4820


def test_negative_population_rejected():
    with pytest.raises(
        ValueError,
        match="population cannot be negative",
    ):
        AdministrativeExposure(
            unit_id="WARD_03",
            unit_name="Ward 3",
            unit_type=(
                AdministrativeUnitType.WARD
            ),
            population=-1,
            road_count=15,
            drain_count=6,
            critical_facility_count=2,
            data_confidence=0.90,
        )


def test_invalid_exposure_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "data_confidence must be between 0 and 1"
        ),
    ):
        AdministrativeExposure(
            unit_id="WARD_03",
            unit_name="Ward 3",
            unit_type=(
                AdministrativeUnitType.WARD
            ),
            population=1000,
            road_count=10,
            drain_count=5,
            critical_facility_count=1,
            data_confidence=1.20,
        )


def test_valid_ward_hazard_inputs():
    hazards = WardHazardInputs(
        flood_risk_score=0.70,
        flood_confidence=0.90,
        landslide_risk_score=0.40,
        landslide_confidence=0.80,
        cascade_intensity_score=0.50,
        cascade_confidence=0.75,
        high_risk_road_count=2,
        overflowing_drain_count=1,
        route_readiness=(
            RouteReadiness.DEGRADED
        ),
    )

    assert hazards.high_risk_road_count == 2


def test_negative_road_count_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "high_risk_road_count cannot be negative"
        ),
    ):
        WardHazardInputs(
            flood_risk_score=0.70,
            flood_confidence=0.90,
            landslide_risk_score=0.40,
            landslide_confidence=0.80,
            cascade_intensity_score=0.50,
            cascade_confidence=0.75,
            high_risk_road_count=-1,
            overflowing_drain_count=1,
            route_readiness=(
                RouteReadiness.DEGRADED
            ),
        )