import pytest

from pravaha_ml.evacuation.models import (
    EvacuationReadinessInputs,
    EvacuationRouteOption,
    EvacuationRouteStatus,
    ShelterOption,
    ShelterStatus,
)


def test_shelter_available_capacity():
    shelter = ShelterOption(
        shelter_id="S_001",
        shelter_name="School Shelter",
        total_capacity=1000,
        current_occupancy=250,
        status=ShelterStatus.AVAILABLE,
        data_confidence=0.90,
    )

    assert (
        shelter.available_capacity
        == 750
    )


def test_unknown_capacity_remains_unknown():
    shelter = ShelterOption(
        shelter_id="S_001",
        shelter_name="School Shelter",
        total_capacity=None,
        current_occupancy=None,
        status=ShelterStatus.UNKNOWN,
        data_confidence=0.50,
    )

    assert (
        shelter.available_capacity
        is None
    )


def test_occupancy_above_capacity_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "current_occupancy cannot exceed total_capacity"
        ),
    ):
        ShelterOption(
            shelter_id="S_001",
            shelter_name="School Shelter",
            total_capacity=100,
            current_occupancy=120,
            status=ShelterStatus.AVAILABLE,
            data_confidence=0.90,
        )


def test_valid_route_option():
    route = EvacuationRouteOption(
        route_id="RT_001",
        shelter_id="S_001",
        route_status=(
            EvacuationRouteStatus.SAFE
        ),
        travel_time_minutes=15.0,
        maximum_risk_score=0.20,
        minimum_confidence=0.90,
    )

    assert (
        route.travel_time_minutes
        == pytest.approx(15.0)
    )


def test_invalid_route_risk_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "maximum_risk_score must be between 0 and 1"
        ),
    ):
        EvacuationRouteOption(
            route_id="RT_001",
            shelter_id="S_001",
            route_status=(
                EvacuationRouteStatus.SAFE
            ),
            travel_time_minutes=15.0,
            maximum_risk_score=1.20,
            minimum_confidence=0.90,
        )


def test_valid_readiness_inputs():
    inputs = EvacuationReadinessInputs(
        unit_id="WARD_03",
        population_exposed=4000,
        current_impact_score=0.70,
        current_impact_confidence=0.90,
        projected_threshold_minutes=40.0,
        projected_threshold_earliest_minutes=30.0,
        trajectory_confidence=0.80,
    )

    assert (
        inputs.population_exposed
        == 4000
    )


def test_negative_threshold_time_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "projected_threshold_minutes cannot be negative"
        ),
    ):
        EvacuationReadinessInputs(
            unit_id="WARD_03",
            population_exposed=4000,
            current_impact_score=0.70,
            current_impact_confidence=0.90,
            projected_threshold_minutes=-5.0,
            projected_threshold_earliest_minutes=None,
            trajectory_confidence=0.80,
        )