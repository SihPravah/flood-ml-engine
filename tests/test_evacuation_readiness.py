import pytest

from pravaha_ml.evacuation.models import (
    EvacuationReadinessInputs,
    EvacuationReadinessLevel,
    EvacuationReadinessPolicy,
    EvacuationRouteOption,
    EvacuationRouteStatus,
    ShelterOption,
    ShelterStatus,
)
from pravaha_ml.evacuation.readiness import (
    assess_evacuation_readiness,
)


def make_inputs(
    **overrides,
) -> EvacuationReadinessInputs:
    values = {
        "unit_id": "WARD_03",
        "population_exposed": 1000,
        "current_impact_score": 0.40,
        "current_impact_confidence": 0.90,
        "projected_threshold_minutes": 90.0,
        "projected_threshold_earliest_minutes": 80.0,
        "trajectory_confidence": 0.85,
        "preparation_buffer_minutes": 10.0,
    }

    values.update(
        overrides
    )

    return EvacuationReadinessInputs(
        **values
    )


def make_shelter(
    shelter_id: str = "S_001",
    *,
    capacity: int = 1500,
    occupancy: int = 100,
    status: ShelterStatus = (
        ShelterStatus.AVAILABLE
    ),
    confidence: float = 0.90,
) -> ShelterOption:
    return ShelterOption(
        shelter_id=shelter_id,
        shelter_name=(
            f"Shelter {shelter_id}"
        ),
        total_capacity=capacity,
        current_occupancy=occupancy,
        status=status,
        data_confidence=confidence,
    )


def make_route(
    route_id: str = "RT_001",
    *,
    shelter_id: str = "S_001",
    status: EvacuationRouteStatus = (
        EvacuationRouteStatus.SAFE
    ),
    travel_time: float = 20.0,
    risk: float = 0.20,
    confidence: float = 0.90,
    authority_closed: bool = False,
) -> EvacuationRouteOption:
    return EvacuationRouteOption(
        route_id=route_id,
        shelter_id=shelter_id,
        route_status=status,
        travel_time_minutes=travel_time,
        maximum_risk_score=risk,
        minimum_confidence=confidence,
        authority_closed=authority_closed,
    )


def test_ready_when_capacity_route_and_time_are_sufficient():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            current_impact_score=0.20
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.READY
    )


def test_short_escalation_window_causes_prepare():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            projected_threshold_minutes=40.0,
            projected_threshold_earliest_minutes=35.0,
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route(
                travel_time=15.0
            )
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.PREPARE
    )

    assert (
        "early_preparation_advised"
        in result.reasons
    )


def test_high_current_impact_causes_prepare():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            current_impact_score=0.70
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.PREPARE
    )


def test_no_reachable_shelter_not_ready():
    result = assess_evacuation_readiness(
        inputs=make_inputs(),
        shelters=[
            make_shelter(
                status=ShelterStatus.CLOSED
            )
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.NOT_READY
    )

    assert (
        "no_reachable_shelter"
        in result.reasons
    )


def test_authority_closed_route_is_not_usable():
    result = assess_evacuation_readiness(
        inputs=make_inputs(),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route(
                authority_closed=True
            )
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.NOT_READY
    )


def test_ai_unsafe_route_is_not_treated_as_available():
    result = assess_evacuation_readiness(
        inputs=make_inputs(),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route(
                status=(
                    EvacuationRouteStatus.UNSAFE
                )
            )
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.NOT_READY
    )


def test_insufficient_capacity_not_ready():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            population_exposed=5000
        ),
        shelters=[
            make_shelter(
                capacity=1000,
                occupancy=0,
            )
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.NOT_READY
    )

    assert (
        "reachable_shelter_capacity_insufficient"
        in result.reasons
    )


def test_partial_capacity_is_degraded():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            population_exposed=2000
        ),
        shelters=[
            make_shelter(
                capacity=1500,
                occupancy=0,
            )
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.DEGRADED
    )


def test_multiple_shelters_capacity_is_combined():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            population_exposed=2000
        ),
        shelters=[
            make_shelter(
                shelter_id="S_001",
                capacity=1200,
                occupancy=100,
            ),
            make_shelter(
                shelter_id="S_002",
                capacity=1200,
                occupancy=100,
            ),
        ],
        routes=[
            make_route(
                route_id="RT_001",
                shelter_id="S_001",
            ),
            make_route(
                route_id="RT_002",
                shelter_id="S_002",
            ),
        ],
    )

    assert (
        result.reachable_known_capacity
        == 2200
    )

    assert (
        result.capacity_ratio
        == pytest.approx(1.10)
    )


def test_route_too_slow_for_projected_window_not_ready():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            projected_threshold_minutes=30.0,
            projected_threshold_earliest_minutes=25.0,
            preparation_buffer_minutes=10.0,
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route(
                travel_time=20.0
            )
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.NOT_READY
    )

    assert (
        "route_time_exceeds_projected_window"
        in result.reasons
    )


def test_earliest_threshold_bound_is_used_conservatively():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            projected_threshold_minutes=60.0,
            projected_threshold_earliest_minutes=40.0,
            preparation_buffer_minutes=10.0,
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route(
                travel_time=15.0
            )
        ],
    )

    assert (
        result.usable_time_window_minutes
        == pytest.approx(30.0)
    )


def test_unknown_population_is_degraded_not_ready_claim():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            population_exposed=None
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.DEGRADED
    )

    assert (
        "population_exposure_unknown"
        in result.reasons
    )


def test_missing_threshold_eta_is_degraded():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            projected_threshold_minutes=None,
            projected_threshold_earliest_minutes=None,
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.DEGRADED
    )

    assert (
        "reliable_hazard_threshold_eta_unavailable"
        in result.reasons
    )


def test_low_confidence_returns_insufficient_data():
    result = assess_evacuation_readiness(
        inputs=make_inputs(
            trajectory_confidence=0.40
        ),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route()
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.INSUFFICIENT_DATA
    )

    assert (
        "evacuation_readiness_confidence_insufficient"
        in result.reasons
    )


def test_low_route_confidence_reduces_overall_confidence():
    result = assess_evacuation_readiness(
        inputs=make_inputs(),
        shelters=[
            make_shelter()
        ],
        routes=[
            make_route(
                confidence=0.50
            )
        ],
    )

    assert (
        result.readiness_level
        == EvacuationReadinessLevel.INSUFFICIENT_DATA
    )


def test_unknown_shelter_reference_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Evacuation route references unknown shelter"
        ),
    ):
        assess_evacuation_readiness(
            inputs=make_inputs(),
            shelters=[
                make_shelter(
                    shelter_id="S_001"
                )
            ],
            routes=[
                make_route(
                    shelter_id="UNKNOWN"
                )
            ],
        )


def test_duplicate_shelters_rejected():
    shelter = make_shelter()

    with pytest.raises(
        ValueError,
        match=(
            "Shelter IDs must be unique"
        ),
    ):
        assess_evacuation_readiness(
            inputs=make_inputs(),
            shelters=[
                shelter,
                shelter,
            ],
            routes=[
                make_route()
            ],
        )


def test_duplicate_routes_rejected():
    route = make_route()

    with pytest.raises(
        ValueError,
        match=(
            "Evacuation route IDs must be unique"
        ),
    ):
        assess_evacuation_readiness(
            inputs=make_inputs(),
            shelters=[
                make_shelter()
            ],
            routes=[
                route,
                route,
            ],
        )


def test_invalid_policy_capacity_order_rejected():
    policy = EvacuationReadinessPolicy(
        minimum_capacity_ratio_ready=0.50,
        minimum_capacity_ratio_degraded=0.80,
    )

    with pytest.raises(
        ValueError,
        match=(
            "minimum_capacity_ratio_degraded cannot exceed"
        ),
    ):
        assess_evacuation_readiness(
            inputs=make_inputs(),
            shelters=[
                make_shelter()
            ],
            routes=[
                make_route()
            ],
            policy=policy,
        )