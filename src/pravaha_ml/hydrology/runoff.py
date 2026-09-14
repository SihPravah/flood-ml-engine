from dataclasses import dataclass


@dataclass(frozen=True)
class RunoffResult:
    rainfall_mm: float
    curve_number: float
    retention_mm: float
    initial_abstraction_mm: float
    runoff_mm: float
    runoff_ratio: float


def validate_curve_number(curve_number: float) -> None:
    """
    Validate SCS Curve Number.

    Standard CN values lie between 1 and 100.
    Higher CN means lower infiltration and higher runoff potential.
    """

    if not 1.0 <= curve_number <= 100.0:
        raise ValueError(
            "curve_number must be between 1 and 100."
        )


def calculate_retention_mm(curve_number: float) -> float:
    """
    Calculate maximum potential retention S in millimetres.

    S = (25400 / CN) - 254
    """

    validate_curve_number(curve_number)

    return (25400.0 / curve_number) - 254.0


def calculate_scs_runoff(
    rainfall_mm: float,
    curve_number: float,
    initial_abstraction_ratio: float = 0.20,
) -> RunoffResult:
    """
    Estimate direct runoff using the SCS Curve Number method.

    Parameters
    ----------
    rainfall_mm:
        Accumulated rainfall depth in millimetres.

        Important:
        This must be accumulated rainfall, NOT rainfall intensity
        in mm/hr.

    curve_number:
        SCS Curve Number in the range 1–100.

    initial_abstraction_ratio:
        Fraction of retention representing interception,
        depression storage and infiltration before runoff starts.

        Classic SCS formulation commonly uses 0.20.

    Returns
    -------
    RunoffResult
    """

    if rainfall_mm < 0.0:
        raise ValueError(
            "rainfall_mm cannot be negative."
        )

    validate_curve_number(curve_number)

    if not 0.0 <= initial_abstraction_ratio <= 1.0:
        raise ValueError(
            "initial_abstraction_ratio must be between 0 and 1."
        )

    retention_mm = calculate_retention_mm(curve_number)

    initial_abstraction_mm = (
        initial_abstraction_ratio * retention_mm
    )

    if rainfall_mm <= initial_abstraction_mm:
        runoff_mm = 0.0
    else:
        numerator = (
            rainfall_mm - initial_abstraction_mm
        ) ** 2

        denominator = (
            rainfall_mm
            - initial_abstraction_mm
            + retention_mm
        )

        runoff_mm = numerator / denominator

    if rainfall_mm == 0.0:
        runoff_ratio = 0.0
    else:
        runoff_ratio = runoff_mm / rainfall_mm

    return RunoffResult(
        rainfall_mm=rainfall_mm,
        curve_number=curve_number,
        retention_mm=retention_mm,
        initial_abstraction_mm=initial_abstraction_mm,
        runoff_mm=runoff_mm,
        runoff_ratio=runoff_ratio,
    )