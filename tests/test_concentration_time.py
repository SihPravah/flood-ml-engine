import pytest

from pravaha_ml.hydrology.concentration_time import (
    calculate_concentration_time,
    calculate_kirpich_time_minutes,
)


def test_kirpich_returns_positive_time():
    result = calculate_kirpich_time_minutes(
        flow_length_m=2000.0,
        slope_fraction=0.10,
    )

    assert result > 0.0


def test_structured_result():
    result = calculate_concentration_time(
        flow_length_m=2000.0,
        slope_fraction=0.10,
    )

    assert result.flow_length_m == 2000.0
    assert result.slope_fraction == 0.10
    assert result.concentration_time_minutes > 0.0
    assert result.method == "KIRPICH"


def test_longer_flow_path_increases_time():
    short = calculate_kirpich_time_minutes(
        flow_length_m=1000.0,
        slope_fraction=0.10,
    )

    long = calculate_kirpich_time_minutes(
        flow_length_m=3000.0,
        slope_fraction=0.10,
    )

    assert long > short


def test_steeper_slope_reduces_time():
    gentle = calculate_kirpich_time_minutes(
        flow_length_m=2000.0,
        slope_fraction=0.05,
    )

    steep = calculate_kirpich_time_minutes(
        flow_length_m=2000.0,
        slope_fraction=0.20,
    )

    assert steep < gentle


def test_zero_flow_length_rejected():
    with pytest.raises(
        ValueError,
        match="flow_length_m must be greater than 0",
    ):
        calculate_kirpich_time_minutes(
            flow_length_m=0.0,
            slope_fraction=0.10,
        )


def test_negative_flow_length_rejected():
    with pytest.raises(
        ValueError,
        match="flow_length_m must be greater than 0",
    ):
        calculate_kirpich_time_minutes(
            flow_length_m=-100.0,
            slope_fraction=0.10,
        )


def test_zero_slope_rejected():
    with pytest.raises(
        ValueError,
        match="slope_fraction must be greater than 0",
    ):
        calculate_kirpich_time_minutes(
            flow_length_m=2000.0,
            slope_fraction=0.0,
        )


def test_negative_slope_rejected():
    with pytest.raises(
        ValueError,
        match="slope_fraction must be greater than 0",
    ):
        calculate_kirpich_time_minutes(
            flow_length_m=2000.0,
            slope_fraction=-0.10,
        )