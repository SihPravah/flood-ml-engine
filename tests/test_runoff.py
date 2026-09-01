import pytest

from pravaha_ml.hydrology.runoff import (
    calculate_retention_mm,
    calculate_scs_runoff,
)


def test_retention_for_curve_number_80():
    retention = calculate_retention_mm(80.0)

    # S = 25400/80 - 254 = 63.5 mm
    assert retention == pytest.approx(63.5)


def test_zero_rainfall_produces_zero_runoff():
    result = calculate_scs_runoff(
        rainfall_mm=0.0,
        curve_number=80.0,
    )

    assert result.runoff_mm == 0.0
    assert result.runoff_ratio == 0.0


def test_rainfall_below_initial_abstraction_produces_zero_runoff():
    result = calculate_scs_runoff(
        rainfall_mm=10.0,
        curve_number=80.0,
    )

    # CN=80:
    # S = 63.5 mm
    # Ia = 0.2 * 63.5 = 12.7 mm
    #
    # Rainfall 10 mm <= Ia, therefore no direct runoff.
    assert result.runoff_mm == 0.0


def test_heavy_rainfall_produces_positive_runoff():
    result = calculate_scs_runoff(
        rainfall_mm=80.0,
        curve_number=80.0,
    )

    assert result.runoff_mm > 0.0
    assert 0.0 < result.runoff_ratio < 1.0


def test_higher_curve_number_produces_more_runoff():
    low_cn = calculate_scs_runoff(
        rainfall_mm=80.0,
        curve_number=60.0,
    )

    high_cn = calculate_scs_runoff(
        rainfall_mm=80.0,
        curve_number=90.0,
    )

    assert high_cn.runoff_mm > low_cn.runoff_mm


def test_negative_rainfall_rejected():
    with pytest.raises(
        ValueError,
        match="rainfall_mm cannot be negative",
    ):
        calculate_scs_runoff(
            rainfall_mm=-5.0,
            curve_number=80.0,
        )


def test_curve_number_above_100_rejected():
    with pytest.raises(
        ValueError,
        match="curve_number must be between 1 and 100",
    ):
        calculate_scs_runoff(
            rainfall_mm=50.0,
            curve_number=120.0,
        )


def test_curve_number_zero_rejected():
    with pytest.raises(
        ValueError,
        match="curve_number must be between 1 and 100",
    ):
        calculate_scs_runoff(
            rainfall_mm=50.0,
            curve_number=0.0,
        )


def test_invalid_initial_abstraction_ratio_rejected():
    with pytest.raises(
        ValueError,
        match="initial_abstraction_ratio must be between 0 and 1",
    ):
        calculate_scs_runoff(
            rainfall_mm=50.0,
            curve_number=80.0,
            initial_abstraction_ratio=1.5,
        )