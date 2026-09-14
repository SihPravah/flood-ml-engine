from dataclasses import dataclass
from enum import Enum

from pravaha_ml.hydrology.runoff import (
    RunoffResult,
    calculate_scs_runoff,
    validate_curve_number,
)


class MoistureCondition(str, Enum):
    """
    Hydrological antecedent moisture condition.

    DRY:
        Catchment has relatively high infiltration capacity.

    NORMAL:
        Base Curve Number conditions.

    WET:
        Catchment is already wet/saturated and therefore
        has higher runoff potential.
    """

    DRY = "DRY"
    NORMAL = "NORMAL"
    WET = "WET"


@dataclass(frozen=True)
class SoilAdjustedRunoffResult:
    base_curve_number: float
    effective_curve_number: float
    moisture_condition: MoistureCondition
    soil_moisture_percentage: float
    runoff: RunoffResult


def classify_soil_moisture(
    soil_moisture_percentage: float,
    dry_threshold_percentage: float,
    wet_threshold_percentage: float,
) -> MoistureCondition:
    """
    Classify soil moisture into DRY, NORMAL or WET.

    Thresholds are intentionally supplied by configuration rather than
    hard-coded globally because appropriate thresholds depend on soil,
    calibration, sensor characteristics and study region.

    Parameters
    ----------
    soil_moisture_percentage:
        Sensor soil-moisture value in the canonical PRAVAHA 0-100 format.

    dry_threshold_percentage:
        Values below this threshold are considered DRY.

    wet_threshold_percentage:
        Values at or above this threshold are considered WET.

    Returns
    -------
    MoistureCondition
    """

    if not 0.0 <= soil_moisture_percentage <= 100.0:
        raise ValueError(
            "soil_moisture_percentage must be between 0 and 100."
        )

    if not 0.0 <= dry_threshold_percentage <= 100.0:
        raise ValueError(
            "dry_threshold_percentage must be between 0 and 100."
        )

    if not 0.0 <= wet_threshold_percentage <= 100.0:
        raise ValueError(
            "wet_threshold_percentage must be between 0 and 100."
        )

    if dry_threshold_percentage >= wet_threshold_percentage:
        raise ValueError(
            "dry_threshold_percentage must be less than "
            "wet_threshold_percentage."
        )

    if soil_moisture_percentage < dry_threshold_percentage:
        return MoistureCondition.DRY

    if soil_moisture_percentage >= wet_threshold_percentage:
        return MoistureCondition.WET

    return MoistureCondition.NORMAL


def adjust_curve_number(
    base_curve_number: float,
    moisture_condition: MoistureCondition,
) -> float:
    """
    Convert the base SCS Curve Number (CN-II) into a moisture-adjusted
    Curve Number.

    Base Curve Number is treated as the normal antecedent-moisture
    condition (CN-II).

    Standard empirical transformations are used for dry (CN-I) and
    wet (CN-III) states.
    """

    validate_curve_number(base_curve_number)

    if moisture_condition == MoistureCondition.NORMAL:
        return base_curve_number

    if moisture_condition == MoistureCondition.DRY:
        adjusted = base_curve_number / (
            2.281 - 0.01281 * base_curve_number
        )

    elif moisture_condition == MoistureCondition.WET:
        adjusted = base_curve_number / (
            0.427 + 0.00573 * base_curve_number
        )

    else:
        raise ValueError(
            f"Unsupported moisture condition: {moisture_condition}"
        )

    return max(1.0, min(adjusted, 100.0))


def calculate_soil_adjusted_runoff(
    rainfall_mm: float,
    base_curve_number: float,
    soil_moisture_percentage: float,
    dry_threshold_percentage: float,
    wet_threshold_percentage: float,
    initial_abstraction_ratio: float = 0.20,
) -> SoilAdjustedRunoffResult:
    """
    Estimate runoff after accounting for current soil-moisture state.

    Pipeline:

        soil moisture
            ↓
        moisture condition
            ↓
        adjusted Curve Number
            ↓
        SCS-CN runoff
    """

    condition = classify_soil_moisture(
        soil_moisture_percentage=soil_moisture_percentage,
        dry_threshold_percentage=dry_threshold_percentage,
        wet_threshold_percentage=wet_threshold_percentage,
    )

    effective_cn = adjust_curve_number(
        base_curve_number=base_curve_number,
        moisture_condition=condition,
    )

    runoff = calculate_scs_runoff(
        rainfall_mm=rainfall_mm,
        curve_number=effective_cn,
        initial_abstraction_ratio=initial_abstraction_ratio,
    )

    return SoilAdjustedRunoffResult(
        base_curve_number=base_curve_number,
        effective_curve_number=effective_cn,
        moisture_condition=condition,
        soil_moisture_percentage=soil_moisture_percentage,
        runoff=runoff,
    )