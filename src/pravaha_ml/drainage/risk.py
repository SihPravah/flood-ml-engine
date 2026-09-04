from math import exp

from pravaha_ml.drainage.capacity import (
    calculate_rectangular_manning_capacity,
)
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainDynamicLoad,
    DrainRiskAssessment,
    DrainRiskLevel,
    DrainStaticProfile,
)


def classify_drain_risk(
    overflow_probability: float,
) -> DrainRiskLevel:
    if not 0.0 <= overflow_probability <= 1.0:
        raise ValueError(
            "overflow_probability must be between 0 and 1."
        )

    if overflow_probability >= 0.85:
        return DrainRiskLevel.SEVERE

    if overflow_probability >= 0.70:
        return DrainRiskLevel.HIGH

    if overflow_probability >= 0.50:
        return DrainRiskLevel.WARNING

    if overflow_probability >= 0.30:
        return DrainRiskLevel.WATCH

    return DrainRiskLevel.LOW


def _sigmoid(
    value: float,
) -> float:
    return (
        1.0
        / (
            1.0
            + exp(-value)
        )
    )


def assess_drain_overflow_risk(
    profile: DrainStaticProfile,
    load: DrainDynamicLoad,
) -> DrainRiskAssessment:
    """
    Estimate drain overload risk from hydraulic capacity and
    dynamic loading.

    The core physical signal is:

        utilization =
            estimated inflow / effective capacity

    A smooth probability mapping is applied around the overload
    region rather than creating a brittle binary cutoff.

    Additional penalties reflect:

        blockage
        uncertain capacity provenance
        poor data confidence

    IMPORTANT:

    This is a transparent decision-support risk score.

    It is not yet a calibrated probability of real-world drain
    failure and must not be represented as such until validated
    against observed overflow/waterlogging events.
    """

    capacity = (
        calculate_rectangular_manning_capacity(
            profile
        )
    )

    if (
        capacity.effective_capacity_m3_per_s
        <= 0.0
    ):
        utilization = float(
            "inf"
        )

        base_probability = 1.0

    else:
        utilization = (
            load.estimated_inflow_m3_per_s
            / capacity.effective_capacity_m3_per_s
        )

        # Smooth transition around 100% capacity.
        base_probability = _sigmoid(
            5.0
            * (
                utilization
                - 1.0
            )
        )

    reasons: list[str] = []

    if utilization >= 1.0:
        reasons.append(
            "estimated_inflow_exceeds_effective_capacity"
        )

    elif utilization >= 0.80:
        reasons.append(
            "drain_near_capacity"
        )

    if profile.blockage_fraction >= 0.50:
        reasons.append(
            "severe_blockage"
        )

    elif profile.blockage_fraction >= 0.20:
        reasons.append(
            "partial_blockage"
        )

    provenance_penalty = 0.0

    if (
        profile.capacity_provenance
        == DrainCapacityProvenance.ESTIMATED
    ):
        provenance_penalty = 0.08

        reasons.append(
            "capacity_estimated"
        )

    elif (
        profile.capacity_provenance
        == DrainCapacityProvenance.DERIVED
    ):
        provenance_penalty = 0.03

        reasons.append(
            "capacity_derived"
        )

    low_confidence_penalty = (
        1.0
        - load.data_confidence
    ) * 0.10

    if load.data_confidence < 0.60:
        reasons.append(
            "dynamic_input_confidence_low"
        )

    blockage_penalty = (
        profile.blockage_fraction
        * 0.10
    )

    overflow_probability = (
        base_probability
        + provenance_penalty
        + low_confidence_penalty
        + blockage_penalty
    )

    overflow_probability = max(
        0.0,
        min(
            overflow_probability,
            1.0,
        ),
    )

    risk_level = classify_drain_risk(
        overflow_probability
    )

    overflow_expected = (
        utilization >= 1.0
    )

    if risk_level in {
        DrainRiskLevel.HIGH,
        DrainRiskLevel.SEVERE,
    }:
        reasons.append(
            "high_drain_overflow_risk"
        )

    return DrainRiskAssessment(
        drain_id=profile.drain_id,
        estimated_inflow_m3_per_s=(
            load.estimated_inflow_m3_per_s
        ),
        effective_capacity_m3_per_s=(
            capacity.effective_capacity_m3_per_s
        ),
        capacity_utilization=float(
            utilization
        ),
        overflow_probability=float(
            overflow_probability
        ),
        risk_level=risk_level,
        data_confidence=(
            load.data_confidence
        ),
        overflow_expected=(
            overflow_expected
        ),
        reasons=tuple(
            reasons
        ),
    )