import pytest

from pravaha_ml.drainage.flow import (
    CatchmentRunoffInput,
    calculate_catchment_discharge,
)


def test_runoff_volume_conversion():
    runoff = CatchmentRunoffInput(
        catchment_id="C_001",
        drain_id="D_001",
        catchment_area_km2=1.0,
        runoff_mm=10.0,
        response_time_minutes=60.0,
        data_confidence=0.90,
    )

    result = calculate_catchment_discharge(
        runoff
    )

    assert (
        result.runoff_volume_m3
        == pytest.approx(
            10_000.0
        )
    )


def test_discharge_conversion():
    runoff = CatchmentRunoffInput(
        catchment_id="C_001",
        drain_id="D_001",
        catchment_area_km2=1.0,
        runoff_mm=10.0,
        response_time_minutes=60.0,
        data_confidence=0.90,
    )

    result = calculate_catchment_discharge(
        runoff
    )

    expected = (
        10_000.0
        / 3600.0
    )

    assert (
        result.characteristic_discharge_m3_per_s
        == pytest.approx(
            expected
        )
    )


def test_larger_catchment_produces_more_discharge():
    small = calculate_catchment_discharge(
        CatchmentRunoffInput(
            catchment_id="C_SMALL",
            drain_id="D_001",
            catchment_area_km2=1.0,
            runoff_mm=10.0,
            response_time_minutes=60.0,
            data_confidence=1.0,
        )
    )

    large = calculate_catchment_discharge(
        CatchmentRunoffInput(
            catchment_id="C_LARGE",
            drain_id="D_001",
            catchment_area_km2=2.0,
            runoff_mm=10.0,
            response_time_minutes=60.0,
            data_confidence=1.0,
        )
    )

    assert (
        large.characteristic_discharge_m3_per_s
        > small.characteristic_discharge_m3_per_s
    )


def test_shorter_response_time_produces_higher_discharge():
    slow = calculate_catchment_discharge(
        CatchmentRunoffInput(
            catchment_id="C_001",
            drain_id="D_001",
            catchment_area_km2=1.0,
            runoff_mm=10.0,
            response_time_minutes=120.0,
            data_confidence=1.0,
        )
    )

    fast = calculate_catchment_discharge(
        CatchmentRunoffInput(
            catchment_id="C_001",
            drain_id="D_001",
            catchment_area_km2=1.0,
            runoff_mm=10.0,
            response_time_minutes=30.0,
            data_confidence=1.0,
        )
    )

    assert (
        fast.characteristic_discharge_m3_per_s
        > slow.characteristic_discharge_m3_per_s
    )


def test_peaking_factor_increases_discharge():
    normal = calculate_catchment_discharge(
        CatchmentRunoffInput(
            catchment_id="C_001",
            drain_id="D_001",
            catchment_area_km2=1.0,
            runoff_mm=10.0,
            response_time_minutes=60.0,
            peaking_factor=1.0,
            data_confidence=1.0,
        )
    )

    peaked = calculate_catchment_discharge(
        CatchmentRunoffInput(
            catchment_id="C_001",
            drain_id="D_001",
            catchment_area_km2=1.0,
            runoff_mm=10.0,
            response_time_minutes=60.0,
            peaking_factor=1.5,
            data_confidence=1.0,
        )
    )

    assert (
        peaked.characteristic_discharge_m3_per_s
        == pytest.approx(
            normal.characteristic_discharge_m3_per_s
            * 1.5
        )
    )


def test_zero_runoff_produces_zero_discharge():
    runoff = CatchmentRunoffInput(
        catchment_id="C_001",
        drain_id="D_001",
        catchment_area_km2=1.0,
        runoff_mm=0.0,
        response_time_minutes=60.0,
        data_confidence=1.0,
    )

    result = calculate_catchment_discharge(
        runoff
    )

    assert (
        result.characteristic_discharge_m3_per_s
        == pytest.approx(0.0)
    )


def test_invalid_area_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "catchment_area_km2 must be greater than 0"
        ),
    ):
        CatchmentRunoffInput(
            catchment_id="C_001",
            drain_id="D_001",
            catchment_area_km2=0.0,
            runoff_mm=10.0,
            response_time_minutes=60.0,
            data_confidence=1.0,
        )


def test_invalid_response_time_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "response_time_minutes must be greater than 0"
        ),
    ):
        CatchmentRunoffInput(
            catchment_id="C_001",
            drain_id="D_001",
            catchment_area_km2=1.0,
            runoff_mm=10.0,
            response_time_minutes=0.0,
            data_confidence=1.0,
        )