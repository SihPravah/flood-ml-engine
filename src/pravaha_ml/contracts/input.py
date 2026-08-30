from datetime import datetime

from pydantic import BaseModel, Field


class Location(BaseModel):
    village: str
    ward: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class SensorMetrics(BaseModel):
    rainfall_mm_per_hr: float = Field(ge=0.0)
    soil_moisture_percentage: float = Field(ge=0.0, le=100.0)
    slope_tilt_degrees: float


class SensorIngestionPayload(BaseModel):
    device_id: str
    timestamp: datetime
    location: Location
    sensor_metrics: SensorMetrics