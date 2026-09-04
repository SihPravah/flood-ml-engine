import pytest

from pravaha_ml.drainage.capacity import (
    calculate_rectangular_manning_capacity,
)
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainStaticProfile,
)


def make_profile(
    *,
    blockage_fraction: float = 0.0,
    condition: DrainCondition = DrainCondition.GOOD,
) -> DrainStaticProfile:
    return DrainStaticProfile(
        drain_id="D_001",
        width_m=2.0,
        depth_m=1.0,
        slope_fraction=0.01,
        manning_roughness=0.015,
        blockage_fraction=blockage_fraction,
        condition=condition,
        capacity_provenance=(
            DrainCapacityProvenance.VERIFIED
        ),
    )


def test_capacity_is_positive():
    result = (
        calculate_rectangular_manning_capacity(
            make_profile()
        )
    )

    assert (
        result.theoretical_capacity_m3_per_s
        > 0.0
    )

    assert (
        result.effective_capacity_m3_per_s
        > 0.0
    )


def test_clear_good_drain_preserves_capacity():
    result = (
        calculate_rectangular_manning_capacity(
            make_profile(
                blockage_fraction=0.0,
                condition=DrainCondition.GOOD,
            )
        )
    )

    assert (
        result.effective_capacity_m3_per_s
        == pytest.approx(
            result.theoretical_capacity_m3_per_s
        )
    )


def test_blockage_reduces_capacity():
    clear = (
        calculate_rectangular_manning_capacity(
            make_profile(
                blockage_fraction=0.0
            )
        )
    )

    blocked = (
        calculate_rectangular_manning_capacity(
            make_profile(
                blockage_fraction=0.50
            )
        )
    )

    assert (
        blocked.effective_capacity_m3_per_s
        < clear.effective_capacity_m3_per_s
    )


def test_poor_condition_reduces_capacity():
    good = (
        calculate_rectangular_manning_capacity(
            make_profile(
                condition=DrainCondition.GOOD
            )
        )
    )

    poor = (
        calculate_rectangular_manning_capacity(
            make_profile(
                condition=DrainCondition.POOR
            )
        )
    )

    assert (
        poor.effective_capacity_m3_per_s
        < good.effective_capacity_m3_per_s
    )


def test_blockage_and_condition_compound():
    good = (
        calculate_rectangular_manning_capacity(
            make_profile()
        )
    )

    degraded = (
        calculate_rectangular_manning_capacity(
            make_profile(
                blockage_fraction=0.40,
                condition=DrainCondition.POOR,
            )
        )
    )

    assert (
        degraded.effective_capacity_m3_per_s
        < good.effective_capacity_m3_per_s
    )