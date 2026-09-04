import pytest

from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainDynamicLoad,
    DrainStaticProfile,
)


def test_valid_drain_profile():
    profile = DrainStaticProfile(
        drain_id="D_001",
        width_m=1.5,
        depth_m=1.0,
        slope_fraction=0.01,
        manning_roughness=0.015,
        blockage_fraction=0.10,
        condition=DrainCondition.GOOD,
        capacity_provenance=(
            DrainCapacityProvenance.VERIFIED
        ),
        catchment_id="C_001",
    )

    assert profile.drain_id == "D_001"


def test_invalid_width_rejected():
    with pytest.raises(
        ValueError,
        match="width_m must be greater than 0",
    ):
        DrainStaticProfile(
            drain_id="D_001",
            width_m=0.0,
            depth_m=1.0,
            slope_fraction=0.01,
            manning_roughness=0.015,
            blockage_fraction=0.0,
            condition=DrainCondition.GOOD,
            capacity_provenance=(
                DrainCapacityProvenance.VERIFIED
            ),
        )


def test_invalid_depth_rejected():
    with pytest.raises(
        ValueError,
        match="depth_m must be greater than 0",
    ):
        DrainStaticProfile(
            drain_id="D_001",
            width_m=1.0,
            depth_m=0.0,
            slope_fraction=0.01,
            manning_roughness=0.015,
            blockage_fraction=0.0,
            condition=DrainCondition.GOOD,
            capacity_provenance=(
                DrainCapacityProvenance.VERIFIED
            ),
        )


def test_invalid_blockage_rejected():
    with pytest.raises(
        ValueError,
        match="blockage_fraction must be between 0 and 1",
    ):
        DrainStaticProfile(
            drain_id="D_001",
            width_m=1.0,
            depth_m=1.0,
            slope_fraction=0.01,
            manning_roughness=0.015,
            blockage_fraction=1.5,
            condition=DrainCondition.GOOD,
            capacity_provenance=(
                DrainCapacityProvenance.VERIFIED
            ),
        )


def test_valid_dynamic_load():
    load = DrainDynamicLoad(
        estimated_inflow_m3_per_s=1.2,
        catchment_runoff_mm=20.0,
        rainfall_mm_per_hr=50.0,
        data_confidence=0.80,
    )

    assert (
        load.estimated_inflow_m3_per_s
        == pytest.approx(1.2)
    )


def test_invalid_dynamic_confidence_rejected():
    with pytest.raises(
        ValueError,
        match="data_confidence must be between 0 and 1",
    ):
        DrainDynamicLoad(
            estimated_inflow_m3_per_s=1.0,
            catchment_runoff_mm=20.0,
            rainfall_mm_per_hr=50.0,
            data_confidence=1.5,
        )