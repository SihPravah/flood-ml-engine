from math import exp

from pravaha_ml.roads.models import (
    RoadFloodAssessment,
    RoadFloodInputs,
    RoadFloodRiskLevel,
    RoadFloodRiskPolicy,
    RoadRecommendation,
)


def _validate_policy(
    policy: RoadFloodRiskPolicy,
) -> None:
    weights = [
        policy.drainage_weight,
        policy.catchment_weight,
        policy.terrain_weight,
        policy.historical_weight,
        policy.stream_weight,
        policy.road_vulnerability_weight,
    ]

    if any(
        weight < 0.0
        for weight in weights
    ):
        raise ValueError(
            "Road flood-risk weights cannot be negative."
        )

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one road flood-risk weight "
            "must be positive."
        )

    if policy.drain_distance_decay_m <= 0.0:
        raise ValueError(
            "drain_distance_decay_m must be greater than 0."
        )

    if not (
        0.0
        <= policy.minimum_passable_confidence
        <= 1.0
    ):
        raise ValueError(
            "minimum_passable_confidence must be "
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
            "Road risk thresholds must be between 0 and 1."
        )

    if not (
        policy.watch_threshold
        <= policy.warning_threshold
        <= policy.high_threshold
        <= policy.severe_threshold
    ):
        raise ValueError(
            "Road risk thresholds must be ordered."
        )


def classify_road_flood_risk(
    risk_score: float,
    policy: RoadFloodRiskPolicy | None = None,
) -> RoadFloodRiskLevel:
    if policy is None:
        policy = RoadFloodRiskPolicy()

    _validate_policy(
        policy
    )

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            "risk_score must be between 0 and 1."
        )

    if risk_score >= policy.severe_threshold:
        return RoadFloodRiskLevel.SEVERE

    if risk_score >= policy.high_threshold:
        return RoadFloodRiskLevel.HIGH

    if risk_score >= policy.warning_threshold:
        return RoadFloodRiskLevel.WARNING

    if risk_score >= policy.watch_threshold:
        return RoadFloodRiskLevel.WATCH

    return RoadFloodRiskLevel.LOW


def _drainage_exposure_score(
    inputs: RoadFloodInputs,
    policy: RoadFloodRiskPolicy,
) -> float:
    """
    Estimate how strongly nearby drainage failure can influence
    this road.

    Drain influence weakens with distance using exponential
    attenuation.

    The physical drainage signal combines:

        overflow probability
        capacity utilization
        actual overflow discharge

    The utilization/discharge transforms are development
    saturation functions rather than calibrated probabilities.
    """

    distance_factor = exp(
        -inputs.nearest_drain_distance_m
        / policy.drain_distance_decay_m
    )

    utilization_score = min(
        inputs.drain_capacity_utilization,
        1.5,
    ) / 1.5

    discharge_score = (
        1.0
        - exp(
            -inputs.drain_overflow_discharge_m3_per_s
        )
    )

    hydraulic_signal = (
        0.50
        * inputs.drain_overflow_probability
        + 0.25
        * utilization_score
        + 0.25
        * discharge_score
    )

    exposure = (
        hydraulic_signal
        * distance_factor
    )

    return max(
        0.0,
        min(
            float(exposure),
            1.0,
        ),
    )


def assess_road_flood_risk(
    inputs: RoadFloodInputs,
    policy: RoadFloodRiskPolicy | None = None,
) -> RoadFloodAssessment:
    """
    Assess flood/waterlogging risk for one road segment.

    The risk score remains separate from data confidence.

    Safety invariant:

        low risk + low confidence != passable

    PRAVAHA may infer AVOID from flood evidence, but CLOSED is
    reserved for authoritative closure information.

    This is currently an explainable development risk index,
    not a calibrated probability of road inundation.
    """

    if policy is None:
        policy = RoadFloodRiskPolicy()

    _validate_policy(
        policy
    )

    drainage_exposure = (
        _drainage_exposure_score(
            inputs,
            policy,
        )
    )

    components = {
        "drainage": (
            drainage_exposure
        ),
        "catchment": (
            inputs.catchment_risk_score
        ),
        "terrain": (
            inputs.terrain_depression_score
        ),
        "historical": (
            inputs.historical_waterlogging_score
        ),
        "stream": (
            inputs.stream_proximity_score
        ),
        "road": (
            inputs.road_surface_vulnerability_score
        ),
    }

    weights = {
        "drainage": (
            policy.drainage_weight
        ),
        "catchment": (
            policy.catchment_weight
        ),
        "terrain": (
            policy.terrain_weight
        ),
        "historical": (
            policy.historical_weight
        ),
        "stream": (
            policy.stream_weight
        ),
        "road": (
            policy.road_vulnerability_weight
        ),
    }

    total_weight = sum(
        weights.values()
    )

    risk_score = sum(
        components[name]
        * weights[name]
        for name in components
    ) / total_weight

    risk_score = max(
        0.0,
        min(
            float(risk_score),
            1.0,
        ),
    )

    risk_level = classify_road_flood_risk(
        risk_score,
        policy,
    )

    reasons: list[str] = []

    if (
        inputs.drain_overflow_probability
        >= 0.70
    ):
        reasons.append(
            "nearby_drain_overflow_risk_high"
        )

    if (
        inputs.drain_capacity_utilization
        >= 1.0
    ):
        reasons.append(
            "nearby_drain_over_capacity"
        )

    if (
        inputs.drain_overflow_discharge_m3_per_s
        > 0.0
    ):
        reasons.append(
            "nearby_drain_overflow_detected"
        )

    if (
        inputs.terrain_depression_score
        >= 0.70
    ):
        reasons.append(
            "road_in_local_depression"
        )

    if (
        inputs.catchment_risk_score
        >= 0.70
    ):
        reasons.append(
            "catchment_flood_risk_high"
        )

    if (
        inputs.historical_waterlogging_score
        >= 0.60
    ):
        reasons.append(
            "historical_waterlogging_evidence"
        )

    if (
        inputs.stream_proximity_score
        >= 0.70
    ):
        reasons.append(
            "high_stream_proximity_exposure"
        )

    if (
        inputs.data_confidence
        < policy.minimum_passable_confidence
    ):
        reasons.append(
            "road_risk_confidence_insufficient"
        )

    authoritative_closure = (
        inputs.authority_closed
    )

    inferred_avoidance = (
        risk_level
        in {
            RoadFloodRiskLevel.HIGH,
            RoadFloodRiskLevel.SEVERE,
        }
    )

    if authoritative_closure:
        recommendation = (
            RoadRecommendation.CLOSED
        )

        reasons.append(
            "authority_confirmed_closure"
        )

    elif inferred_avoidance:
        recommendation = (
            RoadRecommendation.AVOID
        )

        reasons.append(
            "flood_risk_requires_avoidance"
        )

    elif (
        inputs.data_confidence
        < policy.minimum_passable_confidence
    ):
        recommendation = (
            RoadRecommendation.CAUTION
        )

    elif risk_level in {
        RoadFloodRiskLevel.WATCH,
        RoadFloodRiskLevel.WARNING,
    }:
        recommendation = (
            RoadRecommendation.CAUTION
        )

    else:
        recommendation = (
            RoadRecommendation.PASSABLE
        )

    return RoadFloodAssessment(
        road_id=inputs.road_id,
        risk_score=risk_score,
        risk_level=risk_level,
        recommendation=recommendation,
        data_confidence=(
            inputs.data_confidence
        ),
        drainage_exposure_score=(
            drainage_exposure
        ),
        catchment_component=(
            inputs.catchment_risk_score
        ),
        terrain_component=(
            inputs.terrain_depression_score
        ),
        historical_component=(
            inputs.historical_waterlogging_score
        ),
        stream_component=(
            inputs.stream_proximity_score
        ),
        road_vulnerability_component=(
            inputs.road_surface_vulnerability_score
        ),
        nearest_drain_distance_m=(
            inputs.nearest_drain_distance_m
        ),
        inferred_avoidance=(
            inferred_avoidance
        ),
        authoritative_closure=(
            authoritative_closure
        ),
        reasons=tuple(
            reasons
        ),
    )