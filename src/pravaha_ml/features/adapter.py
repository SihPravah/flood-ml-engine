from dataclasses import dataclass

from pravaha_ml.contracts.input import SensorIngestionPayload


@dataclass(frozen=True)
class SensorFeatures:
    device_id: str

    village: str
    ward: str

    latitude: float
    longitude: float

    rainfall_mm_per_hr: float
    soil_saturation: float
    slope_tilt_degrees: float


def build_sensor_features(
    payload: SensorIngestionPayload,
) -> SensorFeatures:
    """
    Convert the canonical PRAVAHA sensor payload into an
    internal representation suitable for ML/hydrology.

    The external DATA_CONTRACT is never modified here.
    """

    metrics = payload.sensor_metrics
    location = payload.location

    soil_saturation = metrics.soil_moisture_percentage / 100.0

    return SensorFeatures(
        device_id=payload.device_id,
        village=location.village,
        ward=location.ward,
        latitude=location.latitude,
        longitude=location.longitude,
        rainfall_mm_per_hr=metrics.rainfall_mm_per_hr,
        soil_saturation=soil_saturation,
        slope_tilt_degrees=metrics.slope_tilt_degrees,
    )