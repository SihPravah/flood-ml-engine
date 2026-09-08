from typing import Iterable

from pravaha_ml.evacuation.models import (
    EvacuationReadinessAssessment,
    EvacuationReadinessInputs,
    EvacuationReadinessLevel,
    EvacuationReadinessPolicy,
    EvacuationRouteOption,
    EvacuationRouteStatus,
    ShelterOption,
    ShelterReadiness,
    ShelterStatus,
)


def _validate_policy(
    policy: EvacuationReadinessPolicy,
) -> None:
    normalized = {
        "minimum_reliable_confidence": (
            policy.minimum_reliable_confidence
        ),
        "safe_route_maximum_risk": (
            policy.safe_route_maximum_risk
        ),
        "safe_route_minimum_confidence": (
            policy.safe_route_minimum_confidence
        ),
    }

    for field_name, value in normalized.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

    if (
        policy.minimum_capacity_ratio_ready
        < 0.0
    ):
        raise ValueError(
            "minimum_capacity_ratio_ready cannot be negative."
        )

    if (
        policy.minimum_capacity_ratio_degraded
        < 0.0
    ):
        raise ValueError(
            "minimum_capacity_ratio_degraded cannot be negative."
        )

    if (
        policy.minimum_capacity_ratio_degraded
        > policy.minimum_capacity_ratio_ready
    ):
        raise ValueError(
            "minimum_capacity_ratio_degraded cannot exceed "
            "minimum_capacity_ratio_ready."
        )

    if policy.urgent_threshold_minutes <= 0.0:
        raise ValueError(
            "urgent_threshold_minutes must be greater than 0."
        )


def _route_is_usable(
    *,
    route: EvacuationRouteOption,
    policy: EvacuationReadinessPolicy,
) -> bool:
    if route.authority_closed:
        return False

    if (
        route.route_status
        in {
            EvacuationRouteStatus.UNSAFE,
            EvacuationRouteStatus.CLOSED,
            EvacuationRouteStatus.UNKNOWN,
        }
    ):
        return False

    if (
        route.maximum_risk_score
        > policy.safe_route_maximum_risk
    ):
        return False

    if (
        route.minimum_confidence
        < policy.safe_route_minimum_confidence
    ):
        return False

    return True


def _best_route_to_shelter(
    *,
    shelter_id: str,
    routes: tuple[
        EvacuationRouteOption,
        ...
    ],
    policy: EvacuationReadinessPolicy,
) -> EvacuationRouteOption | None:
    candidates = [
        route
        for route in routes
        if (
            route.shelter_id
            == shelter_id
            and _route_is_usable(
                route=route,
                policy=policy,
            )
        )
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda route: (
            route.travel_time_minutes,
            route.maximum_risk_score,
        ),
    )


def _assess_shelter(
    *,
    shelter: ShelterOption,
    routes: tuple[
        EvacuationRouteOption,
        ...
    ],
    policy: EvacuationReadinessPolicy,
) -> ShelterReadiness:
    reasons: list[str] = []

    if shelter.status == ShelterStatus.CLOSED:
        reasons.append(
            "shelter_closed"
        )

        return ShelterReadiness(
            shelter_id=shelter.shelter_id,
            shelter_name=shelter.shelter_name,
            available_capacity=(
                shelter.available_capacity
            ),
            reachable=False,
            best_route_id=None,
            best_route_status=None,
            best_route_travel_time_minutes=None,
            reasons=tuple(
                reasons
            ),
        )

    if shelter.status == ShelterStatus.FULL:
        reasons.append(
            "shelter_full"
        )

    if shelter.status == ShelterStatus.UNKNOWN:
        reasons.append(
            "shelter_status_unknown"
        )

    if shelter.available_capacity is None:
        reasons.append(
            "shelter_capacity_unknown"
        )

    best_route = _best_route_to_shelter(
        shelter_id=shelter.shelter_id,
        routes=routes,
        policy=policy,
    )

    reachable = (
        best_route is not None
        and shelter.status
        not in {
            ShelterStatus.CLOSED,
            ShelterStatus.FULL,
            ShelterStatus.UNKNOWN,
        }
    )

    if best_route is None:
        reasons.append(
            "no_usable_route_to_shelter"
        )

    elif (
        best_route.route_status
        == EvacuationRouteStatus.CAUTION
    ):
        reasons.append(
            "best_route_requires_caution"
        )

    if reachable:
        reasons.append(
            "shelter_reachable"
        )

    return ShelterReadiness(
        shelter_id=shelter.shelter_id,
        shelter_name=shelter.shelter_name,
        available_capacity=(
            shelter.available_capacity
        ),
        reachable=reachable,
        best_route_id=(
            best_route.route_id
            if best_route is not None
            else None
        ),
        best_route_status=(
            best_route.route_status
            if best_route is not None
            else None
        ),
        best_route_travel_time_minutes=(
            best_route.travel_time_minutes
            if best_route is not None
            else None
        ),
        reasons=tuple(
            reasons
        ),
    )


def _calculate_usable_time_window(
    inputs: EvacuationReadinessInputs,
) -> float | None:
    """
    Prefer the earliest edge of the projected threshold window.

    That is intentionally more conservative than using only the
    central ETA.
    """

    threshold_time = (
        inputs.projected_threshold_earliest_minutes
    )

    if threshold_time is None:
        threshold_time = (
            inputs.projected_threshold_minutes
        )

    if threshold_time is None:
        return None

    return max(
        threshold_time
        - inputs.preparation_buffer_minutes,
        0.0,
    )


def assess_evacuation_readiness(
    *,
    inputs: EvacuationReadinessInputs,
    shelters: Iterable[
        ShelterOption
    ],
    routes: Iterable[
        EvacuationRouteOption
    ],
    policy: EvacuationReadinessPolicy | None = None,
) -> EvacuationReadinessAssessment:
    """
    Assess whether evacuation preparation appears operationally
    feasible for one ward/village.

    This is DECISION SUPPORT.

    It does not issue an evacuation order.

    The assessment considers:

        exposed population
        available shelter capacity
        route safety
        route confidence
        projected hazard timing
        input confidence

    Important safety principle:

        insufficient evidence != READY
    """

    if policy is None:
        policy = EvacuationReadinessPolicy()

    _validate_policy(
        policy
    )

    shelters = tuple(
        shelters
    )

    routes = tuple(
        routes
    )

    shelter_ids = [
        shelter.shelter_id
        for shelter in shelters
    ]

    if len(
        shelter_ids
    ) != len(
        set(shelter_ids)
    ):
        raise ValueError(
            "Shelter IDs must be unique."
        )

    route_ids = [
        route.route_id
        for route in routes
    ]

    if len(
        route_ids
    ) != len(
        set(route_ids)
    ):
        raise ValueError(
            "Evacuation route IDs must be unique."
        )

    known_shelter_ids = set(
        shelter_ids
    )

    for route in routes:
        if (
            route.shelter_id
            not in known_shelter_ids
        ):
            raise ValueError(
                "Evacuation route references unknown shelter: "
                f"{route.shelter_id}"
            )

    shelter_readiness = tuple(
        _assess_shelter(
            shelter=shelter,
            routes=routes,
            policy=policy,
        )
        for shelter in shelters
    )

    reachable_known_capacity = sum(
        shelter.available_capacity
        for shelter, readiness
        in zip(
            shelters,
            shelter_readiness,
        )
        if (
            readiness.reachable
            and shelter.available_capacity
            is not None
        )
    )

    capacity_ratio: float | None = None

    if inputs.population_exposed is not None:
        if inputs.population_exposed == 0:
            capacity_ratio = 1.0

        else:
            capacity_ratio = (
                reachable_known_capacity
                / inputs.population_exposed
            )

    usable_time_window = (
        _calculate_usable_time_window(
            inputs
        )
    )

    usable_route_times = [
        readiness.best_route_travel_time_minutes
        for readiness in shelter_readiness
        if (
            readiness.reachable
            and readiness.best_route_travel_time_minutes
            is not None
        )
    ]

    fastest_safe_route_minutes = (
        min(
            usable_route_times
        )
        if usable_route_times
        else None
    )

    shelter_confidences = [
        shelter.data_confidence
        for shelter in shelters
        if shelter.status
        != ShelterStatus.CLOSED
    ]

    route_confidences = [
        route.minimum_confidence
        for route in routes
        if not route.authority_closed
    ]

    confidence_components = [
        inputs.current_impact_confidence,
        inputs.trajectory_confidence,
    ]

    confidence_components.extend(
        shelter_confidences
    )

    confidence_components.extend(
        route_confidences
    )

    confidence = min(
        confidence_components
    ) if confidence_components else 0.0

    reasons: list[str] = []

    if (
        inputs.population_exposed
        is None
    ):
        reasons.append(
            "population_exposure_unknown"
        )

    if not shelters:
        reasons.append(
            "no_shelters_configured"
        )

    if not routes:
        reasons.append(
            "no_evacuation_routes_configured"
        )

    reachable_shelter_count = sum(
        readiness.reachable
        for readiness in shelter_readiness
    )

    if reachable_shelter_count == 0:
        reasons.append(
            "no_reachable_shelter"
        )

    if capacity_ratio is not None:
        if (
            capacity_ratio
            >= policy.minimum_capacity_ratio_ready
        ):
            reasons.append(
                "reachable_shelter_capacity_sufficient"
            )

        elif (
            capacity_ratio
            >= policy.minimum_capacity_ratio_degraded
        ):
            reasons.append(
                "reachable_shelter_capacity_partial"
            )

        else:
            reasons.append(
                "reachable_shelter_capacity_insufficient"
            )

    if usable_time_window is not None:
        reasons.append(
            "hazard_threshold_time_window_available"
        )

        if (
            usable_time_window
            <= policy.urgent_threshold_minutes
        ):
            reasons.append(
                "hazard_escalation_window_short"
            )

    else:
        reasons.append(
            "reliable_hazard_threshold_eta_unavailable"
        )

    route_timing_feasible: bool | None = None

    if (
        usable_time_window is not None
        and fastest_safe_route_minutes
        is not None
    ):
        route_timing_feasible = (
            fastest_safe_route_minutes
            < usable_time_window
        )

        if route_timing_feasible:
            reasons.append(
                "safe_route_within_projected_window"
            )

        else:
            reasons.append(
                "route_time_exceeds_projected_window"
            )

    if (
        confidence
        < policy.minimum_reliable_confidence
    ):
        readiness_level = (
            EvacuationReadinessLevel
            .INSUFFICIENT_DATA
        )

        reasons.append(
            "evacuation_readiness_confidence_insufficient"
        )

    elif (
        reachable_shelter_count == 0
        or (
            capacity_ratio is not None
            and capacity_ratio
            < policy.minimum_capacity_ratio_degraded
        )
        or route_timing_feasible is False
    ):
        readiness_level = (
            EvacuationReadinessLevel.NOT_READY
        )

    elif (
        capacity_ratio is None
        or usable_time_window is None
        or fastest_safe_route_minutes is None
    ):
        readiness_level = (
            EvacuationReadinessLevel.DEGRADED
        )

    elif (
        capacity_ratio
        < policy.minimum_capacity_ratio_ready
    ):
        readiness_level = (
            EvacuationReadinessLevel.DEGRADED
        )

    elif (
        usable_time_window
        <= policy.urgent_threshold_minutes
        or inputs.current_impact_score >= 0.50
    ):
        readiness_level = (
            EvacuationReadinessLevel.PREPARE
        )

        reasons.append(
            "early_preparation_advised"
        )

    else:
        readiness_level = (
            EvacuationReadinessLevel.READY
        )

    return EvacuationReadinessAssessment(
        unit_id=inputs.unit_id,
        readiness_level=readiness_level,
        confidence=float(
            confidence
        ),
        population_exposed=(
            inputs.population_exposed
        ),
        reachable_known_capacity=int(
            reachable_known_capacity
        ),
        capacity_ratio=(
            float(capacity_ratio)
            if capacity_ratio is not None
            else None
        ),
        usable_time_window_minutes=(
            float(usable_time_window)
            if usable_time_window is not None
            else None
        ),
        fastest_safe_route_minutes=(
            float(fastest_safe_route_minutes)
            if fastest_safe_route_minutes
            is not None
            else None
        ),
        shelter_readiness=(
            shelter_readiness
        ),
        reasons=tuple(
            reasons
        ),
    )