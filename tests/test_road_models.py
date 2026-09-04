import pytest

from pravaha_ml.roads.models import (
    RoadFloodInputs,
)


def make_inputs(
    **overrides,
) -> RoadFloodInputs:
    values = {
        "road_id": "R_001",
        "catchment_risk_score": 0.40,
        "drain_overflow_probability": 0.30,
        "drain_capacity_utilization": 0.60,
        "drain_overflow_discharge_m3_per_s": 0.0,
        "nearest_drain_distance_m": 25.0,
        "terrain_depression_score": 0.30,
        "stream_proximity_score": 0.20,
        "historical_waterlogging_score": 0.20,
        "road_surface_vulnerability_score": 0.20,
        "data_confidence": 0.90,
        "authority_closed": False,
    }

    values.update(
        overrides
    )

    return RoadFloodInputs(
        **values
    )


def test_valid_road_flood_inputs():
    inputs = make_inputs()

    assert (
        inputs.road_id
        == "R_001"
    )


def test_empty_road_id_rejected():
    with pytest.raises(
        ValueError,
        match="road_id cannot be empty",
    ):
        make_inputs(
            road_id=""
        )


def test_invalid_catchment_score_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "catchment_risk_score must be between 0 and 1"
        ),
    ):
        make_inputs(
            catchment_risk_score=1.20
        )


def test_invalid_drain_probability_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "drain_overflow_probability must be between 0 and 1"
        ),
    ):
        make_inputs(
            drain_overflow_probability=-0.10
        )


def test_negative_drain_utilization_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "drain_capacity_utilization cannot be negative"
        ),
    ):
        make_inputs(
            drain_capacity_utilization=-1.0
        )


def test_negative_overflow_discharge_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "drain_overflow_discharge_m3_per_s cannot be negative"
        ),
    ):
        make_inputs(
            drain_overflow_discharge_m3_per_s=-0.5
        )


def test_negative_drain_distance_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "nearest_drain_distance_m cannot be negative"
        ),
    ):
        make_inputs(
            nearest_drain_distance_m=-10.0
        )


def test_invalid_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "data_confidence must be between 0 and 1"
        ),
    ):
        make_inputs(
            data_confidence=1.50
        )