from math import sqrt

from pravaha_ml.drainage.models import (
    DrainCapacityResult,
    DrainCondition,
    DrainStaticProfile,
)


def _condition_factor(
    condition: DrainCondition,
) -> float:
    """
    Development capacity reduction factors.

    These are not universal engineering constants.

    They provide an explicit placeholder until verified municipal
    inspection/maintenance data can be calibrated.
    """

    if condition == DrainCondition.GOOD:
        return 1.00

    if condition == DrainCondition.DEGRADED:
        return 0.85

    if condition == DrainCondition.POOR:
        return 0.65

    return 0.75


def calculate_rectangular_manning_capacity(
    profile: DrainStaticProfile,
) -> DrainCapacityResult:
    """
    Estimate open-channel drain capacity using Manning's equation
    for a rectangular channel flowing at its supplied effective
    depth.

    Q = (1 / n) * A * R^(2/3) * S^(1/2)

    where:

        Q = discharge capacity in m^3/s
        n = Manning roughness coefficient
        A = cross-sectional flow area
        R = hydraulic radius
        S = longitudinal slope fraction

    Theoretical capacity is subsequently reduced by:

        blockage_fraction
        infrastructure condition factor

    This is a development hydraulic approximation and should not
    be described as a full urban drainage simulation.
    """

    width = profile.width_m
    depth = profile.depth_m

    area_m2 = (
        width * depth
    )

    wetted_perimeter_m = (
        width
        + 2.0 * depth
    )

    hydraulic_radius_m = (
        area_m2
        / wetted_perimeter_m
    )

    theoretical_capacity = (
        (1.0 / profile.manning_roughness)
        * area_m2
        * (
            hydraulic_radius_m
            ** (2.0 / 3.0)
        )
        * sqrt(
            profile.slope_fraction
        )
    )

    blockage_factor = (
        1.0
        - profile.blockage_fraction
    )

    condition_factor = (
        _condition_factor(
            profile.condition
        )
    )

    effective_capacity = (
        theoretical_capacity
        * blockage_factor
        * condition_factor
    )

    return DrainCapacityResult(
        drain_id=profile.drain_id,
        theoretical_capacity_m3_per_s=float(
            theoretical_capacity
        ),
        effective_capacity_m3_per_s=float(
            effective_capacity
        ),
        blockage_reduction_fraction=float(
            profile.blockage_fraction
        ),
        condition_factor=float(
            condition_factor
        ),
    )