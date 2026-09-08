import pytest

from pravaha_ml.evacuation.clearance_models import (
    EvacuationClearanceInputs,
    EvacuationFlowPath,
)


def test_effective_throughput_uses_route_bottleneck():
    path = EvacuationFlowPath(
        path_id="P_001",
        route_id="R_001",
        shelter_id="S_001",
        travel_time_minutes=10.0,
        route_throughput_people_per_minute=50.0,
        shelter_intake_people_per_minute=80.0,
        available_shelter_capacity=1000,
        data_confidence=0.90,
    )

    assert (
        path.effective_throughput_people_per_minute
        == pytest.approx(50.0)
    )


def test_effective_throughput_uses_shelter_bottleneck():
    path = EvacuationFlowPath(
        path_id="P_001",
        route_id="R_001",
        shelter_id="S_001",
        travel_time_minutes=10.0,
        route_throughput_people_per_minute=80.0,
        shelter_intake_people_per_minute=30.0,
        available_shelter_capacity=1000,
        data_confidence=0.90,
    )

    assert (
        path.effective_throughput_people_per_minute
        == pytest.approx(30.0)
    )


def test_invalid_route_throughput_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "route_throughput_people_per_minute "
            "must be greater than 0"
        ),
    ):
        EvacuationFlowPath(
            path_id="P_001",
            route_id="R_001",
            shelter_id="S_001",
            travel_time_minutes=10.0,
            route_throughput_people_per_minute=0.0,
            shelter_intake_people_per_minute=30.0,
            available_shelter_capacity=1000,
            data_confidence=0.90,
        )


def test_negative_capacity_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "available_shelter_capacity cannot be negative"
        ),
    ):
        EvacuationFlowPath(
            path_id="P_001",
            route_id="R_001",
            shelter_id="S_001",
            travel_time_minutes=10.0,
            route_throughput_people_per_minute=50.0,
            shelter_intake_people_per_minute=30.0,
            available_shelter_capacity=-1,
            data_confidence=0.90,
        )


def test_valid_clearance_inputs():
    inputs = EvacuationClearanceInputs(
        unit_id="WARD_03",
        population_to_move=4000,
        mobilization_delay_minutes=10.0,
        hazard_window_minutes=60.0,
        data_confidence=0.90,
    )

    assert (
        inputs.population_to_move
        == 4000
    )


def test_negative_population_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "population_to_move cannot be negative"
        ),
    ):
        EvacuationClearanceInputs(
            unit_id="WARD_03",
            population_to_move=-1,
            mobilization_delay_minutes=10.0,
            hazard_window_minutes=60.0,
            data_confidence=0.90,
        )


def test_negative_hazard_window_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "hazard_window_minutes cannot be negative"
        ),
    ):
        EvacuationClearanceInputs(
            unit_id="WARD_03",
            population_to_move=1000,
            mobilization_delay_minutes=10.0,
            hazard_window_minutes=-5.0,
            data_confidence=0.90,
        )