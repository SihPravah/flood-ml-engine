import pytest

from pravaha_ml.roads.models import (
    RoadFloodInputs,
    RoadFloodRiskLevel,
    RoadFloodRiskPolicy,
    RoadRecommendation,
)
from pravaha_ml.roads.risk import (
    assess_road_flood_risk,
    classify_road_flood_risk,
)


def make_inputs(
    **overrides,
) -> RoadFloodInputs:
    values = {
        "road_id": "R_001",
        "catchment_risk_score": 0.10,
        "drain_overflow_probability": 0.10,
        "drain_capacity_utilization": 0.20,
        "drain_overflow_discharge_m3_per_s": 0.0,
        "nearest_drain_distance_m": 100.0,
        "terrain_depression_score": 0.10,
        "stream_proximity_score": 0.10,
        "historical_waterlogging_score": 0.10,
        "road_surface_vulnerability_score": 0.10,
        "data_confidence": 0.95,
        "authority_closed": False,
    }

    values.update(
        overrides
    )

    return RoadFloodInputs(
        **values
    )


def test_low_risk_high_confidence_can_be_passable():
    result = assess_road_flood_risk(
        make_inputs()
    )

    assert (
        result.risk_level
        == RoadFloodRiskLevel.LOW
    )

    assert (
        result.recommendation
        == RoadRecommendation.PASSABLE
    )


def test_low_risk_low_confidence_is_not_passable():
    result = assess_road_flood_risk(
        make_inputs(
            data_confidence=0.30
        )
    )

    assert (
        result.recommendation
        == RoadRecommendation.CAUTION
    )

    assert (
        "road_risk_confidence_insufficient"
        in result.reasons
    )


def test_severe_inputs_recommend_avoid():
    result = assess_road_flood_risk(
        make_inputs(
            catchment_risk_score=0.95,
            drain_overflow_probability=0.98,
            drain_capacity_utilization=1.50,
            drain_overflow_discharge_m3_per_s=4.0,
            nearest_drain_distance_m=2.0,
            terrain_depression_score=0.95,
            stream_proximity_score=0.90,
            historical_waterlogging_score=0.95,
            road_surface_vulnerability_score=0.90,
        )
    )

    assert (
        result.risk_level
        in {
            RoadFloodRiskLevel.HIGH,
            RoadFloodRiskLevel.SEVERE,
        }
    )

    assert (
        result.recommendation
        == RoadRecommendation.AVOID
    )

    assert (
        result.inferred_avoidance
        is True
    )


def test_authoritative_closure_overrides_risk():
    result = assess_road_flood_risk(
        make_inputs(
            authority_closed=True
        )
    )

    assert (
        result.recommendation
        == RoadRecommendation.CLOSED
    )

    assert (
        result.authoritative_closure
        is True
    )

    assert (
        "authority_confirmed_closure"
        in result.reasons
    )


def test_model_does_not_call_inferred_avoidance_closed():
    result = assess_road_flood_risk(
        make_inputs(
            catchment_risk_score=1.0,
            drain_overflow_probability=1.0,
            drain_capacity_utilization=2.0,
            drain_overflow_discharge_m3_per_s=5.0,
            nearest_drain_distance_m=0.0,
            terrain_depression_score=1.0,
            stream_proximity_score=1.0,
            historical_waterlogging_score=1.0,
            road_surface_vulnerability_score=1.0,
            authority_closed=False,
        )
    )

    assert (
        result.recommendation
        != RoadRecommendation.CLOSED
    )

    assert (
        result.recommendation
        == RoadRecommendation.AVOID
    )


def test_nearer_overflowing_drain_increases_road_risk():
    near = assess_road_flood_risk(
        make_inputs(
            drain_overflow_probability=0.90,
            drain_capacity_utilization=1.20,
            drain_overflow_discharge_m3_per_s=1.5,
            nearest_drain_distance_m=5.0,
        )
    )

    far = assess_road_flood_risk(
        make_inputs(
            drain_overflow_probability=0.90,
            drain_capacity_utilization=1.20,
            drain_overflow_discharge_m3_per_s=1.5,
            nearest_drain_distance_m=500.0,
        )
    )

    assert (
        near.drainage_exposure_score
        > far.drainage_exposure_score
    )

    assert (
        near.risk_score
        > far.risk_score
    )


def test_local_depression_increases_risk():
    flat = assess_road_flood_risk(
        make_inputs(
            terrain_depression_score=0.10
        )
    )

    depressed = assess_road_flood_risk(
        make_inputs(
            terrain_depression_score=0.95
        )
    )

    assert (
        depressed.risk_score
        > flat.risk_score
    )


def test_historical_waterlogging_increases_risk():
    low_history = (
        assess_road_flood_risk(
            make_inputs(
                historical_waterlogging_score=0.0
            )
        )
    )

    high_history = (
        assess_road_flood_risk(
            make_inputs(
                historical_waterlogging_score=1.0
            )
        )
    )

    assert (
        high_history.risk_score
        > low_history.risk_score
    )


def test_high_drain_utilization_reason_is_exposed():
    result = assess_road_flood_risk(
        make_inputs(
            drain_capacity_utilization=1.20,
        )
    )

    assert (
        "nearby_drain_over_capacity"
        in result.reasons
    )


def test_overflow_reason_is_exposed():
    result = assess_road_flood_risk(
        make_inputs(
            drain_overflow_discharge_m3_per_s=0.80,
        )
    )

    assert (
        "nearby_drain_overflow_detected"
        in result.reasons
    )


def test_risk_classification_boundaries():
    assert (
        classify_road_flood_risk(0.10)
        == RoadFloodRiskLevel.LOW
    )

    assert (
        classify_road_flood_risk(0.35)
        == RoadFloodRiskLevel.WATCH
    )

    assert (
        classify_road_flood_risk(0.55)
        == RoadFloodRiskLevel.WARNING
    )

    assert (
        classify_road_flood_risk(0.75)
        == RoadFloodRiskLevel.HIGH
    )

    assert (
        classify_road_flood_risk(0.90)
        == RoadFloodRiskLevel.SEVERE
    )


def test_invalid_risk_score_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "risk_score must be between 0 and 1"
        ),
    ):
        classify_road_flood_risk(
            1.50
        )


def test_invalid_policy_weight_rejected():
    policy = RoadFloodRiskPolicy(
        drainage_weight=-1.0
    )

    with pytest.raises(
        ValueError,
        match=(
            "Road flood-risk weights cannot be negative"
        ),
    ):
        assess_road_flood_risk(
            make_inputs(),
            policy=policy,
        )


def test_invalid_threshold_order_rejected():
    policy = RoadFloodRiskPolicy(
        watch_threshold=0.60,
        warning_threshold=0.40,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Road risk thresholds must be ordered"
        ),
    ):
        assess_road_flood_risk(
            make_inputs(),
            policy=policy,
        )