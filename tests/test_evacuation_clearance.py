import pytest

from pravaha_ml.evacuation.clearance import (
    assess_evacuation_clearance,
)
from pravaha_ml.evacuation.clearance_models import (
    ClearanceFeasibility,
    ClearanceMarginStatus,
    EvacuationClearanceInputs,
    EvacuationClearancePolicy,
    EvacuationFlowPath,
)


def make_inputs(
    **overrides,
) -> EvacuationClearanceInputs:
    values = {
        "unit_id": "WARD_03",
        "population_to_move": 1000,
        "mobilization_delay_minutes": 10.0,
        "hazard_window_minutes": 90.0,
        "data_confidence": 0.90,
    }

    values.update(
        overrides
    )

    return EvacuationClearanceInputs(
        **values
    )


def make_path(
    path_id: str = "P_001",
    *,
    route_id: str = "R_001",
    shelter_id: str = "S_001",
    travel_time: float = 10.0,
    route_throughput: float = 50.0,
    shelter_throughput: float = 50.0,
    capacity: int = 1500,
    confidence: float = 0.90,
    usable: bool = True,
    authority_closed: bool = False,
) -> EvacuationFlowPath:
    return EvacuationFlowPath(
        path_id=path_id,
        route_id=route_id,
        shelter_id=shelter_id,
        travel_time_minutes=travel_time,
        route_throughput_people_per_minute=(
            route_throughput
        ),
        shelter_intake_people_per_minute=(
            shelter_throughput
        ),
        available_shelter_capacity=capacity,
        data_confidence=confidence,
        usable=usable,
        authority_closed=authority_closed,
    )


def test_single_path_clearance_time():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            population_to_move=1000,
        ),
        paths=[
            make_path(
                travel_time=10.0,
                route_throughput=50.0,
                shelter_throughput=50.0,
            )
        ],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.FEASIBLE
    )

    # 10 min mobilization
    # + 10 min travel
    # + 1000 / 50 = 20 min flow
    # ≈ 40 min total
    assert (
        result.estimated_clearance_minutes
        == pytest.approx(
            40.0,
            abs=0.05,
        )
    )


def test_parallel_paths_reduce_clearance_time():
    single = assess_evacuation_clearance(
        inputs=make_inputs(
            population_to_move=1000,
        ),
        paths=[
            make_path()
        ],
    )

    parallel = assess_evacuation_clearance(
        inputs=make_inputs(
            population_to_move=1000,
        ),
        paths=[
            make_path(
                path_id="P_001",
                shelter_id="S_001",
                capacity=1000,
            ),
            make_path(
                path_id="P_002",
                route_id="R_002",
                shelter_id="S_002",
                capacity=1000,
            ),
        ],
    )

    assert (
        parallel.estimated_clearance_minutes
        <
        single.estimated_clearance_minutes
    )


def test_capacity_insufficient_detected():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            population_to_move=2000,
        ),
        paths=[
            make_path(
                capacity=1000
            )
        ],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.CAPACITY_INSUFFICIENT
    )

    assert (
        result.estimated_clearance_minutes
        is None
    )


def test_closed_path_is_excluded():
    result = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path(
                authority_closed=True
            )
        ],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.NO_USABLE_PATH
    )


def test_unusable_path_is_excluded():
    result = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path(
                usable=False
            )
        ],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.NO_USABLE_PATH
    )


def test_sufficient_margin_identified():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            hazard_window_minutes=90.0
        ),
        paths=[
            make_path()
        ],
    )

    assert (
        result.margin_status
        == ClearanceMarginStatus.SUFFICIENT
    )

    assert (
        result.clearance_margin_minutes
        > 0.0
    )


def test_tight_margin_identified():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            hazard_window_minutes=50.0
        ),
        paths=[
            make_path()
        ],
    )

    assert (
        result.margin_status
        == ClearanceMarginStatus.TIGHT
    )


def test_insufficient_margin_identified():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            hazard_window_minutes=30.0
        ),
        paths=[
            make_path()
        ],
    )

    assert (
        result.margin_status
        == ClearanceMarginStatus.INSUFFICIENT
    )

    assert (
        "clearance_time_exceeds_hazard_window"
        in result.reasons
    )


def test_unknown_hazard_window_preserved():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            hazard_window_minutes=None
        ),
        paths=[
            make_path()
        ],
    )

    assert (
        result.margin_status
        == ClearanceMarginStatus.UNKNOWN
    )

    assert (
        result.clearance_margin_minutes
        is None
    )

    assert (
        "hazard_window_unknown"
        in result.reasons
    )


def test_low_confidence_blocks_reliable_clearance():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            data_confidence=0.40
        ),
        paths=[
            make_path()
        ],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.INSUFFICIENT_DATA
    )


def test_low_path_confidence_blocks_reliable_clearance():
    result = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path(
                confidence=0.40
            )
        ],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.INSUFFICIENT_DATA
    )


def test_mobilization_delay_increases_clearance_time():
    fast_start = assess_evacuation_clearance(
        inputs=make_inputs(
            mobilization_delay_minutes=5.0
        ),
        paths=[
            make_path()
        ],
    )

    slow_start = assess_evacuation_clearance(
        inputs=make_inputs(
            mobilization_delay_minutes=20.0
        ),
        paths=[
            make_path()
        ],
    )

    assert (
        slow_start.estimated_clearance_minutes
        >
        fast_start.estimated_clearance_minutes
    )


def test_shelter_intake_can_be_bottleneck():
    fast_intake = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path(
                route_throughput=100.0,
                shelter_throughput=100.0,
            )
        ],
    )

    slow_intake = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path(
                route_throughput=100.0,
                shelter_throughput=20.0,
            )
        ],
    )

    assert (
        slow_intake.estimated_clearance_minutes
        >
        fast_intake.estimated_clearance_minutes
    )


def test_zero_population_clears_immediately():
    result = assess_evacuation_clearance(
        inputs=make_inputs(
            population_to_move=0
        ),
        paths=[],
    )

    assert (
        result.feasibility
        == ClearanceFeasibility.FEASIBLE
    )

    assert (
        result.estimated_clearance_minutes
        == pytest.approx(0.0)
    )


def test_path_contributions_are_reported():
    result = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path()
        ],
    )

    assert len(
        result.path_contributions
    ) == 1

    assert (
        result.path_contributions[
            0
        ].estimated_people_moved
        >= 999
    )


def test_duplicate_path_ids_rejected():
    path = make_path()

    with pytest.raises(
        ValueError,
        match=(
            "Evacuation flow path IDs must be unique"
        ),
    ):
        assess_evacuation_clearance(
            inputs=make_inputs(),
            paths=[
                path,
                path,
            ],
        )


def test_invalid_margin_policy_rejected():
    policy = EvacuationClearancePolicy(
        sufficient_margin_minutes=5.0,
        tight_margin_minutes=10.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "tight_margin_minutes cannot exceed "
            "sufficient_margin_minutes"
        ),
    ):
        assess_evacuation_clearance(
            inputs=make_inputs(),
            paths=[
                make_path()
            ],
            policy=policy,
        )


def test_planning_estimate_reason_is_exposed():
    result = assess_evacuation_clearance(
        inputs=make_inputs(),
        paths=[
            make_path()
        ],
    )

    assert (
        "clearance_time_is_planning_estimate"
        in result.reasons
    )