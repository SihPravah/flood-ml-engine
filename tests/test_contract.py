import pytest
from pydantic import ValidationError

from pravaha_ml.contracts.input import SensorIngestionPayload


VALID_PAYLOAD = {
    "device_id": "SIM_NODE_04",
    "timestamp": "2026-08-30T14:30:00Z",
    "location": {
        "village": "Munnar",
        "ward": "Ward_3",
        "latitude": 10.0889,
        "longitude": 77.0595,
    },
    "sensor_metrics": {
        "rainfall_mm_per_hr": 45.5,
        "soil_moisture_percentage": 82.0,
        "slope_tilt_degrees": 12.2,
    },
}


def test_valid_sensor_payload():
    payload = SensorIngestionPayload.model_validate(VALID_PAYLOAD)

    assert payload.device_id == "SIM_NODE_04"
    assert payload.location.village == "Munnar"
    assert payload.location.ward == "Ward_3"

    assert payload.sensor_metrics.rainfall_mm_per_hr == 45.5
    assert payload.sensor_metrics.soil_moisture_percentage == 82.0
    assert payload.sensor_metrics.slope_tilt_degrees == 12.2


def test_invalid_soil_moisture_rejected():
    invalid_payload = {
        **VALID_PAYLOAD,
        "sensor_metrics": {
            **VALID_PAYLOAD["sensor_metrics"],
            "soil_moisture_percentage": 120.0,
        },
    }

    with pytest.raises(ValidationError):
        SensorIngestionPayload.model_validate(invalid_payload)


def test_invalid_latitude_rejected():
    invalid_payload = {
        **VALID_PAYLOAD,
        "location": {
            **VALID_PAYLOAD["location"],
            "latitude": 120.0,
        },
    }

    with pytest.raises(ValidationError):
        SensorIngestionPayload.model_validate(invalid_payload)


def test_negative_rainfall_rejected():
    invalid_payload = {
        **VALID_PAYLOAD,
        "sensor_metrics": {
            **VALID_PAYLOAD["sensor_metrics"],
            "rainfall_mm_per_hr": -10.0,
        },
    }

    with pytest.raises(ValidationError):
        SensorIngestionPayload.model_validate(invalid_payload)