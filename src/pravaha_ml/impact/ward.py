from pravaha_ml.impact.models import (
    AdministrativeExposure,
    ImpactLevel,
    RouteReadiness,
    WardHazardInputs,
    WardImpactAssessment,
    WardImpactPolicy,
)


def _validate_policy(
    policy: WardImpactPolicy,
) -> None:
    weights = [
        policy.flood_weight,
        policy.landslide_weight,
        policy.cascade_weight,
        policy.infrastructure_weight,
    ]

    if any(
        weight < 0.0
        for weight in weights
    ):
        raise ValueError(
            "Ward impact weights cannot be negative."
        )

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one ward impact weight must be positive."
        )

    if not (
        0.0
        <= policy.minimum_reliable_confidence
        <= 1.0
    ):
        raise ValueError(
            "minimum_reliable_confidence must be "
            "between 0 and 1."
        )

    thresholds = [
        policy.watch_threshold,
        policy.warning_threshold,
        policy.high_threshold,
        policy.severe_threshold,
    ]

    if any(
        not 0.0 <= threshold <= 1.0
        for threshold in thresholds
    ):
        raise ValueError(
            "Ward impact thresholds must be between 0 and 1."
        )

    if not (
        policy.watch_threshold
        <= policy.warning_threshold
        <= policy.high_threshold
        <= policy.severe_threshold
    ):
        raise ValueError(
            "Ward impact thresholds must be ordered."
        )


def _classify_impact(
    score: float,
    policy: WardImpactPolicy,
) -> ImpactLevel:
    if score >= policy.severe_threshold:
        return ImpactLevel.SEVERE

    if score >= policy.high_threshold:
        return ImpactLevel.HIGH

    if score >= policy.warning_threshold:
        return ImpactLevel.WARNING

    if score >= policy.watch_threshold:
        return ImpactLevel.WATCH

    return ImpactLevel.LOW


def _infrastructure_pressure(
    *,
    hazards: WardHazardInputs,
    exposure: AdministrativeExposure,
) -> float:
    """
    Approximate infrastructure disruption pressure.

    This uses normalized counts and route readiness.

    It is an explicit decision-support index, not a probability.
    """

    road_pressure = min(
        hazards.high_risk_road_count
        / max(
            exposure.road_count,
            1,
        ),
        1.0,
    )

    drain_pressure = min(
        hazards.overflowing_drain_count
        / max(
            exposure.drain_count,
            1,
        ),
        1.0,
    )

    if hazards.route_readiness == RouteReadiness.UNSAFE:
        route_pressure = 1.0

    elif hazards.route_readiness == RouteReadiness.DEGRADED:
        route_pressure = 0.70

    elif hazards.route_readiness == RouteReadiness.UNKNOWN:
        route_pressure = 0.50

    else:
        route_pressure = 0.0

    return float(
        0.40 * road_pressure
        + 0.35 * drain_pressure
        + 0.25 * route_pressure
    )


def assess_ward_impact(
    *,
    exposure: AdministrativeExposure,
    hazards: WardHazardInputs,
    policy: WardImpactPolicy | None = None,
) -> WardImpactAssessment:
    """
    Produce one hyper-local ward/village impact assessment.

    Hazard severity and confidence remain separate.

    Low hazard score with insufficient confidence is NOT promoted
    to a reliable low-impact interpretation.
    """

    if policy is None:
        policy = WardImpactPolicy()

    _validate_policy(
        policy
    )

    infrastructure = _infrastructure_pressure(
        hazards=hazards,
        exposure=exposure,
    )

    total_weight = (
        policy.flood_weight
        + policy.landslide_weight
        + policy.cascade_weight
        + policy.infrastructure_weight
    )

    impact_score = (
        hazards.flood_risk_score
        * policy.flood_weight
        + hazards.landslide_risk_score
        * policy.landslide_weight
        + hazards.cascade_intensity_score
        * policy.cascade_weight
        + infrastructure
        * policy.infrastructure_weight
    ) / total_weight

    impact_score = max(
        0.0,
        min(
            float(impact_score),
            1.0,
        ),
    )

    confidence = min(
        exposure.data_confidence,
        hazards.flood_confidence,
        hazards.landslide_confidence,
        hazards.cascade_confidence,
    )

    reliable = (
        confidence
        >= policy.minimum_reliable_confidence
    )

    if reliable:
        impact_level = _classify_impact(
            impact_score,
            policy,
        )
    else:
        impact_level = (
            ImpactLevel.INSUFFICIENT_DATA
        )

    reasons: list[str] = []

    if hazards.flood_risk_score >= 0.70:
        reasons.append(
            "high_flash_flood_risk"
        )

    if hazards.landslide_risk_score >= 0.70:
        reasons.append(
            "high_landslide_susceptibility"
        )

    if hazards.cascade_intensity_score >= 0.60:
        reasons.append(
            "significant_cascade_potential"
        )

    if hazards.overflowing_drain_count > 0:
        reasons.append(
            "drainage_overflow_present"
        )

    if hazards.high_risk_road_count > 0:
        reasons.append(
            "high_risk_roads_present"
        )

    if hazards.route_readiness == RouteReadiness.DEGRADED:
        reasons.append(
            "evacuation_route_degraded"
        )

    elif hazards.route_readiness == RouteReadiness.UNSAFE:
        reasons.append(
            "evacuation_route_unsafe"
        )

    elif hazards.route_readiness == RouteReadiness.UNKNOWN:
        reasons.append(
            "evacuation_route_status_unknown"
        )

    if exposure.population is None:
        reasons.append(
            "population_exposure_unknown"
        )

    if not reliable:
        reasons.append(
            "ward_impact_confidence_insufficient"
        )

    return WardImpactAssessment(
        unit_id=exposure.unit_id,
        unit_name=exposure.unit_name,
        unit_type=exposure.unit_type,
        impact_score=impact_score,
        impact_level=impact_level,
        confidence=float(
            confidence
        ),
        reliable=reliable,
        population_exposed=(
            exposure.population
        ),
        high_risk_road_count=(
            hazards.high_risk_road_count
        ),
        overflowing_drain_count=(
            hazards.overflowing_drain_count
        ),
        critical_facility_count=(
            exposure.critical_facility_count
        ),
        route_readiness=(
            hazards.route_readiness
        ),
        reasons=tuple(
            reasons
        ),
    )