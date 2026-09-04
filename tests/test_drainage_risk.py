import pytest

from pravaha_ml.drainage.capacity import (
    calculate_rectangular_manning_capacity,
)
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainDynamicLoad,
    DrainRiskLevel,
    DrainStaticProfile,
)
from pravaha_ml.drainage.risk import (
    assess_drain_overflow_risk,
    classify_drain_risk,
)


def make_profile(
    *,
    blockage_fraction: float = 0.0,
    provenance: DrainCapacityProvenance = (
        DrainCapacityProvenance.VERIFIED
    ),
) -> DrainStaticProfile:
    return DrainStaticProfile(
        drain_id="D_001",
        width_m=2.0,
        depth_m=1.0,
        slope_fraction=0.01,
        manning_roughness=0.015,
        blockage_fraction=blockage_fraction,
        condition=DrainCondition.GOOD,
        capacity_provenance=provenance,
    )


def test_low_load_has_lower_risk():
    profile = make_profile()

    capacity = (
        calculate_rectangular_manning_capacity(
            profile
        )
    )

    load = DrainDynamicLoad(
        estimated_inflow_m3_per_s=(
            capacity.effective_capacity_m3_per_s
            * 0.20
        ),
        catchment_runoff_mm=5.0,
        rainfall_mm_per_hr=15.0,
        data_confidence=1.0,
    )

    result = assess_drain_overflow_risk(
        profile=profile,
        load=load,
    )

    assert (
        result.capacity_utilization
        == pytest.approx(0.20)
    )

    assert (
        result.risk_level
        in {
            DrainRiskLevel.LOW,
            DrainRiskLevel.WATCH,
        }
    )

    assert result.overflow_expected is False


def test_over_capacity_is_flagged():
    profile = make_profile()

    capacity = (
        calculate_rectangular_manning_capacity(
            profile
        )
    )

    load = DrainDynamicLoad(
        estimated_inflow_m3_per_s=(
            capacity.effective_capacity_m3_per_s
            * 1.25
        ),
        catchment_runoff_mm=40.0,
        rainfall_mm_per_hr=80.0,
        data_confidence=0.90,
    )

    result = assess_drain_overflow_risk(
        profile=profile,
        load=load,
    )

    assert (
        result.capacity_utilization
        > 1.0
    )

    assert result.overflow_expected is True

    assert (
        "estimated_inflow_exceeds_effective_capacity"
        in result.reasons
    )


def test_more_inflow_increases_overflow_probability():
    profile = make_profile()

    capacity = (
        calculate_rectangular_manning_capacity(
            profile
        )
    )

    low = assess_drain_overflow_risk(
        profile=profile,
        load=DrainDynamicLoad(
            estimated_inflow_m3_per_s=(
                capacity.effective_capacity_m3_per_s
                * 0.40
            ),
            catchment_runoff_mm=10.0,
            rainfall_mm_per_hr=30.0,
            data_confidence=1.0,
        ),
    )

    high = assess_drain_overflow_risk(
        profile=profile,
        load=DrainDynamicLoad(
            estimated_inflow_m3_per_s=(
                capacity.effective_capacity_m3_per_s
                * 1.20
            ),
            catchment_runoff_mm=40.0,
            rainfall_mm_per_hr=90.0,
            data_confidence=1.0,
        ),
    )

    assert (
        high.overflow_probability
        > low.overflow_probability
    )


def test_blockage_increases_risk():
    clear = make_profile(
        blockage_fraction=0.0
    )

    blocked = make_profile(
        blockage_fraction=0.50
    )

    load = DrainDynamicLoad(
        estimated_inflow_m3_per_s=2.0,
        catchment_runoff_mm=20.0,
        rainfall_mm_per_hr=50.0,
        data_confidence=1.0,
    )

    clear_result = (
        assess_drain_overflow_risk(
            profile=clear,
            load=load,
        )
    )

    blocked_result = (
        assess_drain_overflow_risk(
            profile=blocked,
            load=load,
        )
    )

    assert (
        blocked_result.overflow_probability
        > clear_result.overflow_probability
    )


def test_estimated_capacity_adds_uncertainty_reason():
    profile = make_profile(
        provenance=(
            DrainCapacityProvenance.ESTIMATED
        )
    )

    load = DrainDynamicLoad(
        estimated_inflow_m3_per_s=1.0,
        catchment_runoff_mm=10.0,
        rainfall_mm_per_hr=30.0,
        data_confidence=0.90,
    )

    result = assess_drain_overflow_risk(
        profile=profile,
        load=load,
    )

    assert (
        "capacity_estimated"
        in result.reasons
    )


def test_low_dynamic_confidence_is_flagged():
    profile = make_profile()

    load = DrainDynamicLoad(
        estimated_inflow_m3_per_s=1.0,
        catchment_runoff_mm=10.0,
        rainfall_mm_per_hr=30.0,
        data_confidence=0.40,
    )

    result = assess_drain_overflow_risk(
        profile=profile,
        load=load,
    )

    assert (
        "dynamic_input_confidence_low"
        in result.reasons
    )


def test_risk_classification_boundaries():
    assert (
        classify_drain_risk(0.10)
        == DrainRiskLevel.LOW
    )

    assert (
        classify_drain_risk(0.35)
        == DrainRiskLevel.WATCH
    )

    assert (
        classify_drain_risk(0.55)
        == DrainRiskLevel.WARNING
    )

    assert (
        classify_drain_risk(0.75)
        == DrainRiskLevel.HIGH
    )

    assert (
        classify_drain_risk(0.90)
        == DrainRiskLevel.SEVERE
    )


def test_invalid_probability_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "overflow_probability must be between 0 and 1"
        ),
    ):
        classify_drain_risk(
            1.20
        )