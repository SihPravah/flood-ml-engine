from pravaha_ml.landslide.models import (
    LandslideAssessment,
    LandslideDataProvenance,
    LandslideInputs,
    LandslideRiskLevel,
    LandslideRiskPolicy,
)


def _validate_policy(
    policy: LandslideRiskPolicy,
) -> None:
    weights = [
        policy.slope_weight,
        policy.soil_weight,
        policy.short_rain_weight,
        policy.antecedent_rain_weight,
        policy.historical_weight,
        policy.land_cover_weight,
    ]

    if any(
        weight < 0.0
        for weight in weights
    ):
        raise ValueError(
            "Landslide susceptibility weights "
            "cannot be negative."
        )

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one landslide susceptibility "
            "weight must be positive."
        )

    positive_references = {
        "slope_reference_degrees": (
            policy.slope_reference_degrees
        ),
        "rain_1h_reference_mm": (
            policy.rain_1h_reference_mm
        ),
        "rain_24h_reference_mm": (
            policy.rain_24h_reference_mm
        ),
        "api_reference_mm": (
            policy.api_reference_mm
        ),
    }

    for field_name, value in positive_references.items():
        if value <= 0.0:
            raise ValueError(
                f"{field_name} must be greater than 0."
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
            "Landslide risk thresholds must be "
            "between 0 and 1."
        )

    if not (
        policy.watch_threshold
        <= policy.warning_threshold
        <= policy.high_threshold
        <= policy.severe_threshold
    ):
        raise ValueError(
            "Landslide risk thresholds must be ordered."
        )


def _bounded_ratio(
    value: float,
    reference: float,
) -> float:
    return max(
        0.0,
        min(
            value / reference,
            1.0,
        ),
    )


def classify_landslide_risk(
    susceptibility_score: float,
    policy: LandslideRiskPolicy | None = None,
) -> LandslideRiskLevel:
    if policy is None:
        policy = LandslideRiskPolicy()

    _validate_policy(
        policy
    )

    if not 0.0 <= susceptibility_score <= 1.0:
        raise ValueError(
            "susceptibility_score must be between 0 and 1."
        )

    if (
        susceptibility_score
        >= policy.severe_threshold
    ):
        return LandslideRiskLevel.SEVERE

    if (
        susceptibility_score
        >= policy.high_threshold
    ):
        return LandslideRiskLevel.HIGH

    if (
        susceptibility_score
        >= policy.warning_threshold
    ):
        return LandslideRiskLevel.WARNING

    if (
        susceptibility_score
        >= policy.watch_threshold
    ):
        return LandslideRiskLevel.WATCH

    return LandslideRiskLevel.LOW


def assess_landslide_susceptibility(
    inputs: LandslideInputs,
    policy: LandslideRiskPolicy | None = None,
) -> LandslideAssessment:
    """
    Estimate rainfall-triggered landslide susceptibility.

    The score combines:

        terrain slope
        soil wetness
        recent rainfall
        antecedent rainfall
        historical landslide evidence
        land-cover/disturbance evidence

    This is an explainable development susceptibility index.

    It is NOT yet a calibrated probability that a landslide will
    occur, and it must not be represented as a confirmed event.
    """

    if policy is None:
        policy = LandslideRiskPolicy()

    _validate_policy(
        policy
    )

    slope_component = _bounded_ratio(
        inputs.slope_degrees,
        policy.slope_reference_degrees,
    )

    soil_component = (
        inputs.soil_saturation
    )

    short_rain_component = _bounded_ratio(
        inputs.rain_1h_mm,
        policy.rain_1h_reference_mm,
    )

    rain_24h_component = _bounded_ratio(
        inputs.rain_24h_mm,
        policy.rain_24h_reference_mm,
    )

    api_component = _bounded_ratio(
        inputs.api_mm,
        policy.api_reference_mm,
    )

    antecedent_rain_component = (
        0.60 * rain_24h_component
        + 0.40 * api_component
    )

    historical_component = (
        inputs.historical_landslide_score
    )

    land_cover_component = (
        inputs.land_cover_disturbance_score
    )

    components = {
        "slope": slope_component,
        "soil": soil_component,
        "short_rain": short_rain_component,
        "antecedent_rain": (
            antecedent_rain_component
        ),
        "historical": historical_component,
        "land_cover": land_cover_component,
    }

    weights = {
        "slope": policy.slope_weight,
        "soil": policy.soil_weight,
        "short_rain": (
            policy.short_rain_weight
        ),
        "antecedent_rain": (
            policy.antecedent_rain_weight
        ),
        "historical": (
            policy.historical_weight
        ),
        "land_cover": (
            policy.land_cover_weight
        ),
    }

    total_weight = sum(
        weights.values()
    )

    susceptibility_score = sum(
        components[name]
        * weights[name]
        for name in components
    ) / total_weight

    susceptibility_score = max(
        0.0,
        min(
            float(susceptibility_score),
            1.0,
        ),
    )

    confidence = inputs.data_confidence

    reasons: list[str] = []

    if inputs.slope_degrees >= 30.0:
        reasons.append(
            "steep_terrain"
        )

    if inputs.soil_saturation >= 0.75:
        reasons.append(
            "high_soil_saturation"
        )

    if inputs.rain_1h_mm >= 50.0:
        reasons.append(
            "intense_recent_rainfall"
        )

    if (
        inputs.rain_24h_mm >= 120.0
        or inputs.api_mm >= 100.0
    ):
        reasons.append(
            "high_antecedent_rainfall"
        )

    if (
        inputs.historical_landslide_score
        >= 0.60
    ):
        reasons.append(
            "historical_landslide_evidence"
        )

    if (
        inputs.land_cover_disturbance_score
        >= 0.60
    ):
        reasons.append(
            "land_cover_or_surface_disturbance"
        )

    if (
        inputs.historical_inventory_provenance
        == LandslideDataProvenance.ESTIMATED
    ):
        confidence *= 0.90

        reasons.append(
            "historical_inventory_estimated"
        )

    elif (
        inputs.historical_inventory_provenance
        == LandslideDataProvenance.MISSING
    ):
        confidence *= 0.70

        reasons.append(
            "historical_inventory_missing"
        )

    confidence = max(
        0.0,
        min(
            float(confidence),
            1.0,
        ),
    )

    reliable = (
        confidence
        >= policy.minimum_reliable_confidence
    )

    if not reliable:
        reasons.append(
            "landslide_assessment_low_confidence"
        )

    risk_level = classify_landslide_risk(
        susceptibility_score,
        policy,
    )

    return LandslideAssessment(
        zone_id=inputs.zone_id,
        susceptibility_score=(
            susceptibility_score
        ),
        risk_level=risk_level,
        confidence=confidence,
        reliable=reliable,
        slope_component=float(
            slope_component
        ),
        soil_component=float(
            soil_component
        ),
        short_rain_component=float(
            short_rain_component
        ),
        antecedent_rain_component=float(
            antecedent_rain_component
        ),
        historical_component=float(
            historical_component
        ),
        land_cover_component=float(
            land_cover_component
        ),
        reasons=tuple(
            reasons
        ),
    )