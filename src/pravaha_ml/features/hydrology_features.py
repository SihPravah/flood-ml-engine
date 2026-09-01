from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from pravaha_ml.features.antecedent import build_antecedent_features
from pravaha_ml.features.rainfall import (
    RainfallObservation,
    build_rainfall_features,
)
from pravaha_ml.hydrology.concentration_time import (
    calculate_concentration_time,
)
from pravaha_ml.hydrology.soil_state import (
    calculate_soil_adjusted_runoff,
)


@dataclass(frozen=True)
class HydrologyFeatures:
    # Rainfall accumulation features
    rain_15m_mm: float
    rain_30m_mm: float
    rain_1h_mm: float
    rain_3h_mm: float
    rain_6h_mm: float
    rain_24h_mm: float

    # Antecedent rainfall state
    api_mm: float

    # Soil state
    soil_moisture_percentage: float
    soil_saturation: float
    moisture_condition: str

    # Curve Number / runoff
    base_curve_number: float
    effective_curve_number: float
    runoff_mm: float
    runoff_ratio: float

    # Catchment response
    flow_length_m: float
    slope_fraction: float
    concentration_time_minutes: float


def build_hydrology_features(
    rainfall_observations: Iterable[RainfallObservation],
    prediction_time: datetime,
    soil_moisture_percentage: float,
    base_curve_number: float,
    flow_length_m: float,
    slope_fraction: float,
    dry_threshold_percentage: float,
    wet_threshold_percentage: float,
    api_decay_factor: float = 0.90,
) -> HydrologyFeatures:
    """
    Build a complete hydrology feature vector for one catchment
    at one prediction time.

    This function integrates:

    - temporal rainfall accumulation
    - antecedent precipitation
    - soil moisture state
    - moisture-adjusted Curve Number
    - SCS-CN runoff
    - time of concentration
    """

    rainfall_features = build_rainfall_features(
        observations=rainfall_observations,
        prediction_time=prediction_time,
    )

    antecedent_features = build_antecedent_features(
        observations=rainfall_observations,
        prediction_time=prediction_time,
        decay_factor=api_decay_factor,
    )

    soil_result = calculate_soil_adjusted_runoff(
        rainfall_mm=rainfall_features.rain_1h_mm,
        base_curve_number=base_curve_number,
        soil_moisture_percentage=soil_moisture_percentage,
        dry_threshold_percentage=dry_threshold_percentage,
        wet_threshold_percentage=wet_threshold_percentage,
    )

    concentration_result = calculate_concentration_time(
        flow_length_m=flow_length_m,
        slope_fraction=slope_fraction,
    )

    soil_saturation = soil_moisture_percentage / 100.0

    return HydrologyFeatures(
        rain_15m_mm=rainfall_features.rain_15m_mm,
        rain_30m_mm=rainfall_features.rain_30m_mm,
        rain_1h_mm=rainfall_features.rain_1h_mm,
        rain_3h_mm=rainfall_features.rain_3h_mm,
        rain_6h_mm=rainfall_features.rain_6h_mm,
        rain_24h_mm=rainfall_features.rain_24h_mm,
        api_mm=antecedent_features.api_mm,
        soil_moisture_percentage=soil_moisture_percentage,
        soil_saturation=soil_saturation,
        moisture_condition=soil_result.moisture_condition.value,
        base_curve_number=base_curve_number,
        effective_curve_number=soil_result.effective_curve_number,
        runoff_mm=soil_result.runoff.runoff_mm,
        runoff_ratio=soil_result.runoff.runoff_ratio,
        flow_length_m=flow_length_m,
        slope_fraction=slope_fraction,
        concentration_time_minutes=(
            concentration_result.concentration_time_minutes
        ),
    )