from typing import Iterable

from pravaha_ml.evacuation.clearance_models import (
    ClearanceFeasibility,
    ClearanceMarginStatus,
    EvacuationClearanceAssessment,
    EvacuationClearanceInputs,
    EvacuationClearancePolicy,
    EvacuationFlowPath,
    PathClearanceContribution,
)


def _validate_policy(
    policy: EvacuationClearancePolicy,
) -> None:
    if not (
        0.0
        <= policy.minimum_reliable_confidence
        <= 1.0
    ):
        raise ValueError(
            "minimum_reliable_confidence must be "
            "between 0 and 1."
        )

    if policy.sufficient_margin_minutes < 0.0:
        raise ValueError(
            "sufficient_margin_minutes cannot be negative."
        )

    if policy.tight_margin_minutes < 0.0:
        raise ValueError(
            "tight_margin_minutes cannot be negative."
        )

    if (
        policy.tight_margin_minutes
        > policy.sufficient_margin_minutes
    ):
        raise ValueError(
            "tight_margin_minutes cannot exceed "
            "sufficient_margin_minutes."
        )

    if policy.search_tolerance_minutes <= 0.0:
        raise ValueError(
            "search_tolerance_minutes must be greater than 0."
        )

    if policy.maximum_search_minutes <= 0.0:
        raise ValueError(
            "maximum_search_minutes must be greater than 0."
        )


def _usable_paths(
    paths: Iterable[
        EvacuationFlowPath
    ],
) -> tuple[
    EvacuationFlowPath,
    ...
]:
    return tuple(
        path
        for path in paths
        if (
            path.usable
            and not path.authority_closed
            and path.available_shelter_capacity > 0
        )
    )


def _people_delivered_by_time(
    *,
    path: EvacuationFlowPath,
    elapsed_minutes: float,
    mobilization_delay_minutes: float,
) -> float:
    """
    Approximate how many people can complete this path by time T.

    Flow begins after mobilization.

    People then require route travel time before they count as
    delivered to the shelter.

    After arrival begins, throughput is limited by:

        min(route throughput, shelter intake throughput)

    and by remaining shelter capacity.
    """

    usable_flow_time = (
        elapsed_minutes
        - mobilization_delay_minutes
        - path.travel_time_minutes
    )

    if usable_flow_time <= 0.0:
        return 0.0

    flow_capacity = (
        path.effective_throughput_people_per_minute
        * usable_flow_time
    )

    return min(
        flow_capacity,
        float(
            path.available_shelter_capacity
        ),
    )


def _total_people_delivered(
    *,
    paths: tuple[
        EvacuationFlowPath,
        ...
    ],
    elapsed_minutes: float,
    mobilization_delay_minutes: float,
) -> float:
    return sum(
        _people_delivered_by_time(
            path=path,
            elapsed_minutes=elapsed_minutes,
            mobilization_delay_minutes=(
                mobilization_delay_minutes
            ),
        )
        for path in paths
    )


def _estimate_clearance_time(
    *,
    inputs: EvacuationClearanceInputs,
    paths: tuple[
        EvacuationFlowPath,
        ...
    ],
    policy: EvacuationClearancePolicy,
) -> float | None:
    """
    Find the earliest time when all required evacuees could
    theoretically reach available shelter capacity.

    Parallel routes operate simultaneously.

    Binary search is used because multiple paths may have
    different:
        travel times
        throughputs
        shelter capacities
    """

    if inputs.population_to_move == 0:
        return 0.0

    maximum_deliverable = _total_people_delivered(
        paths=paths,
        elapsed_minutes=(
            policy.maximum_search_minutes
        ),
        mobilization_delay_minutes=(
            inputs.mobilization_delay_minutes
        ),
    )

    if (
        maximum_deliverable
        + 1e-9
        < inputs.population_to_move
    ):
        return None

    lower = 0.0
    upper = (
        policy.maximum_search_minutes
    )

    while (
        upper - lower
        > policy.search_tolerance_minutes
    ):
        midpoint = (
            lower + upper
        ) / 2.0

        delivered = _total_people_delivered(
            paths=paths,
            elapsed_minutes=midpoint,
            mobilization_delay_minutes=(
                inputs.mobilization_delay_minutes
            ),
        )

        if (
            delivered
            >= inputs.population_to_move
        ):
            upper = midpoint
        else:
            lower = midpoint

    return float(
        upper
    )


def _classify_margin(
    *,
    margin_minutes: float | None,
    policy: EvacuationClearancePolicy,
) -> ClearanceMarginStatus:
    if margin_minutes is None:
        return (
            ClearanceMarginStatus.UNKNOWN
        )

    if margin_minutes < 0.0:
        return (
            ClearanceMarginStatus.INSUFFICIENT
        )

    if (
        margin_minutes
        < policy.sufficient_margin_minutes
    ):
        return (
            ClearanceMarginStatus.TIGHT
        )

    return (
        ClearanceMarginStatus.SUFFICIENT
    )


def _path_contributions(
    *,
    paths: tuple[
        EvacuationFlowPath,
        ...
    ],
    clearance_time_minutes: float,
    mobilization_delay_minutes: float,
) -> tuple[
    PathClearanceContribution,
    ...
]:
    contributions = []

    for path in paths:
        moved = _people_delivered_by_time(
            path=path,
            elapsed_minutes=(
                clearance_time_minutes
            ),
            mobilization_delay_minutes=(
                mobilization_delay_minutes
            ),
        )

        contributions.append(
            PathClearanceContribution(
                path_id=path.path_id,
                route_id=path.route_id,
                shelter_id=path.shelter_id,
                effective_throughput_people_per_minute=(
                    path
                    .effective_throughput_people_per_minute
                ),
                available_capacity=(
                    path.available_shelter_capacity
                ),
                estimated_people_moved=int(
                    round(
                        moved
                    )
                ),
            )
        )

    return tuple(
        contributions
    )


def assess_evacuation_clearance(
    *,
    inputs: EvacuationClearanceInputs,
    paths: Iterable[
        EvacuationFlowPath
    ],
    policy: EvacuationClearancePolicy | None = None,
) -> EvacuationClearanceAssessment:
    """
    Estimate evacuation clearance time and compare it with the
    available hazard window.

    IMPORTANT:

    This is a simplified planning model.

    It does not simulate individual agents, traffic interactions,
    panic, pedestrian behavior, vehicle queues, bridge bottlenecks,
    or emergency-service interference.

    Therefore:

        estimated_clearance_minutes

    must be presented as a planning estimate, not a guaranteed
    evacuation completion time.
    """

    if policy is None:
        policy = EvacuationClearancePolicy()

    _validate_policy(
        policy
    )

    paths = tuple(
        paths
    )

    path_ids = [
        path.path_id
        for path in paths
    ]

    if len(
        path_ids
    ) != len(
        set(path_ids)
    ):
        raise ValueError(
            "Evacuation flow path IDs must be unique."
        )

    usable_paths = _usable_paths(
        paths
    )

    reasons: list[str] = []

    reachable_capacity = sum(
        path.available_shelter_capacity
        for path in usable_paths
    )

    confidence_values = [
        inputs.data_confidence
    ]

    confidence_values.extend(
        path.data_confidence
        for path in usable_paths
    )

    confidence = min(
        confidence_values
    ) if confidence_values else 0.0

    if (
        confidence
        < policy.minimum_reliable_confidence
    ):
        reasons.append(
            "clearance_estimate_confidence_insufficient"
        )

        return EvacuationClearanceAssessment(
            unit_id=inputs.unit_id,
            feasibility=(
                ClearanceFeasibility
                .INSUFFICIENT_DATA
            ),
            estimated_clearance_minutes=None,
            hazard_window_minutes=(
                inputs.hazard_window_minutes
            ),
            clearance_margin_minutes=None,
            margin_status=(
                ClearanceMarginStatus.UNKNOWN
            ),
            population_to_move=(
                inputs.population_to_move
            ),
            reachable_capacity=int(
                reachable_capacity
            ),
            confidence=float(
                confidence
            ),
            path_contributions=(),
            reasons=tuple(
                reasons
            ),
        )

    if (
        inputs.population_to_move == 0
    ):
        reasons.append(
            "no_population_requires_movement"
        )

        return EvacuationClearanceAssessment(
            unit_id=inputs.unit_id,
            feasibility=(
                ClearanceFeasibility.FEASIBLE
            ),
            estimated_clearance_minutes=0.0,
            hazard_window_minutes=(
                inputs.hazard_window_minutes
            ),
            clearance_margin_minutes=(
                inputs.hazard_window_minutes
                if inputs.hazard_window_minutes
                is not None
                else None
            ),
            margin_status=_classify_margin(
                margin_minutes=(
                    inputs.hazard_window_minutes
                    if inputs.hazard_window_minutes
                    is not None
                    else None
                ),
                policy=policy,
            ),
            population_to_move=0,
            reachable_capacity=int(
                reachable_capacity
            ),
            confidence=float(
                confidence
            ),
            path_contributions=(),
            reasons=tuple(
                reasons
            ),
        )

    if not usable_paths:
        reasons.append(
            "no_usable_evacuation_flow_path"
        )

        return EvacuationClearanceAssessment(
            unit_id=inputs.unit_id,
            feasibility=(
                ClearanceFeasibility.NO_USABLE_PATH
            ),
            estimated_clearance_minutes=None,
            hazard_window_minutes=(
                inputs.hazard_window_minutes
            ),
            clearance_margin_minutes=None,
            margin_status=(
                ClearanceMarginStatus.UNKNOWN
            ),
            population_to_move=(
                inputs.population_to_move
            ),
            reachable_capacity=0,
            confidence=float(
                confidence
            ),
            path_contributions=(),
            reasons=tuple(
                reasons
            ),
        )

    if (
        reachable_capacity
        < inputs.population_to_move
    ):
        reasons.append(
            "reachable_shelter_capacity_below_population"
        )

        return EvacuationClearanceAssessment(
            unit_id=inputs.unit_id,
            feasibility=(
                ClearanceFeasibility
                .CAPACITY_INSUFFICIENT
            ),
            estimated_clearance_minutes=None,
            hazard_window_minutes=(
                inputs.hazard_window_minutes
            ),
            clearance_margin_minutes=None,
            margin_status=(
                ClearanceMarginStatus.UNKNOWN
            ),
            population_to_move=(
                inputs.population_to_move
            ),
            reachable_capacity=int(
                reachable_capacity
            ),
            confidence=float(
                confidence
            ),
            path_contributions=(),
            reasons=tuple(
                reasons
            ),
        )

    clearance_time = (
        _estimate_clearance_time(
            inputs=inputs,
            paths=usable_paths,
            policy=policy,
        )
    )

    if clearance_time is None:
        reasons.append(
            "population_not_clearable_within_search_horizon"
        )

        return EvacuationClearanceAssessment(
            unit_id=inputs.unit_id,
            feasibility=(
                ClearanceFeasibility
                .CAPACITY_INSUFFICIENT
            ),
            estimated_clearance_minutes=None,
            hazard_window_minutes=(
                inputs.hazard_window_minutes
            ),
            clearance_margin_minutes=None,
            margin_status=(
                ClearanceMarginStatus.UNKNOWN
            ),
            population_to_move=(
                inputs.population_to_move
            ),
            reachable_capacity=int(
                reachable_capacity
            ),
            confidence=float(
                confidence
            ),
            path_contributions=(),
            reasons=tuple(
                reasons
            ),
        )

    margin_minutes = None

    if (
        inputs.hazard_window_minutes
        is not None
    ):
        margin_minutes = (
            inputs.hazard_window_minutes
            - clearance_time
        )

    margin_status = _classify_margin(
        margin_minutes=margin_minutes,
        policy=policy,
    )

    if (
        inputs.hazard_window_minutes
        is None
    ):
        reasons.append(
            "hazard_window_unknown"
        )

    elif (
        margin_status
        == ClearanceMarginStatus.INSUFFICIENT
    ):
        reasons.append(
            "clearance_time_exceeds_hazard_window"
        )

    elif (
        margin_status
        == ClearanceMarginStatus.TIGHT
    ):
        reasons.append(
            "evacuation_clearance_margin_tight"
        )

    else:
        reasons.append(
            "evacuation_clearance_margin_sufficient"
        )

    reasons.append(
        "clearance_time_is_planning_estimate"
    )

    contributions = _path_contributions(
        paths=usable_paths,
        clearance_time_minutes=(
            clearance_time
        ),
        mobilization_delay_minutes=(
            inputs.mobilization_delay_minutes
        ),
    )

    return EvacuationClearanceAssessment(
        unit_id=inputs.unit_id,
        feasibility=(
            ClearanceFeasibility.FEASIBLE
        ),
        estimated_clearance_minutes=float(
            clearance_time
        ),
        hazard_window_minutes=(
            inputs.hazard_window_minutes
        ),
        clearance_margin_minutes=(
            float(
                margin_minutes
            )
            if margin_minutes is not None
            else None
        ),
        margin_status=margin_status,
        population_to_move=(
            inputs.population_to_move
        ),
        reachable_capacity=int(
            reachable_capacity
        ),
        confidence=float(
            confidence
        ),
        path_contributions=(
            contributions
        ),
        reasons=tuple(
            reasons
        ),
    )