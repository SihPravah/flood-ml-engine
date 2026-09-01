from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ConcentrationTimeResult:
    flow_length_m: float
    slope_fraction: float
    concentration_time_minutes: float
    method: str


def validate_inputs(
    flow_length_m: float,
    slope_fraction: float,
) -> None:
    if flow_length_m <= 0.0:
        raise ValueError(
            "flow_length_m must be greater than 0."
        )

    if slope_fraction <= 0.0:
        raise ValueError(
            "slope_fraction must be greater than 0."
        )


def calculate_kirpich_time_minutes(
    flow_length_m: float,
    slope_fraction: float,
) -> float:
    """
    Estimate time of concentration using the Kirpich equation.

    Formula:

        Tc = 0.0195 * L^0.77 * S^-0.385

    where:

        Tc = time of concentration in minutes
        L  = longest flow path in metres
        S  = dimensionless slope fraction

    Important:
    slope_fraction is not degrees.

    Example:
        10% slope -> 0.10
    """

    validate_inputs(
        flow_length_m=flow_length_m,
        slope_fraction=slope_fraction,
    )

    concentration_time = (
        0.0195
        * math.pow(flow_length_m, 0.77)
        * math.pow(slope_fraction, -0.385)
    )

    return concentration_time


def calculate_concentration_time(
    flow_length_m: float,
    slope_fraction: float,
) -> ConcentrationTimeResult:
    """
    Return structured catchment response-time information.
    """

    concentration_time = calculate_kirpich_time_minutes(
        flow_length_m=flow_length_m,
        slope_fraction=slope_fraction,
    )

    return ConcentrationTimeResult(
        flow_length_m=flow_length_m,
        slope_fraction=slope_fraction,
        concentration_time_minutes=concentration_time,
        method="KIRPICH",
    )