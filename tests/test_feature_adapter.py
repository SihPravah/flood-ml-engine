from pravaha_ml.contracts.input import SensorIngestionPayload
from pravaha_ml.features.adapter import build_sensor_features


def test_build_sensor_features():
    raw_payload = {
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

    payload = SensorIngestionPayload.model_validate(raw_payload)

    features = build_sensor_features(payload)

    assert features.device_id == "SIM_NODE_04"

    assert features.village == "Munnar"
    assert features.ward == "Ward_3"

    assert features.latitude == 10.0889
    assert features.longitude == 77.0595

    assert features.rainfall_mm_per_hr == 45.5

    # 82% becomes 0.82 internally
    assert features.soil_saturation == 0.82

    assert features.slope_tilt_degrees == 12.2