from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


@dataclass(frozen=True)
class RainfallObservation:
    """
    A single rainfall intensity observation.

    rainfall_mm_per_hr is an intensity, not an accumulated depth.
    """

    timestamp: datetime
    rainfall_mm_per_hr: float


@dataclass(frozen=True)
class RainfallFeatures:
    rain_15m_mm: float
    rain_30m_mm: float
    rain_1h_mm: float
    rain_3h_mm: float
    rain_6h_mm: float
    rain_24h_mm: float


def _validate_observations(
    observations: Iterable[RainfallObservation],
) -> list[RainfallObservation]:
    observations = list(observations)

    for obs in observations:
        if obs.rainfall_mm_per_hr < 0:
            raise ValueError("Rainfall intensity cannot be negative.")

        if obs.timestamp.tzinfo is None:
            raise ValueError(
                "Rainfall observation timestamps must be timezone-aware."
            )

    return sorted(
        observations,
        key=lambda obs: obs.timestamp,
    )


def _accumulate_window(
    observations: list[RainfallObservation],
    end_time: datetime,
    window: timedelta,
) -> float:
    """
    Estimate accumulated rainfall depth over a time window.

    Each observation is treated as the rainfall intensity applying until
    the next observation.

    For the final observation, intensity is applied until end_time.

    Depth contribution:

        depth_mm = intensity_mm_per_hr * duration_hours
    """

    if not observations:
        return 0.0

    start_time = end_time - window

    total_mm = 0.0

    for index, observation in enumerate(observations):
        interval_start = observation.timestamp

        if index + 1 < len(observations):
            interval_end = observations[index + 1].timestamp
        else:
            interval_end = end_time

        overlap_start = max(interval_start, start_time)
        overlap_end = min(interval_end, end_time)

        if overlap_end <= overlap_start:
            continue

        duration_hours = (
            overlap_end - overlap_start
        ).total_seconds() / 3600.0

        total_mm += (
            observation.rainfall_mm_per_hr
            * duration_hours
        )

    return total_mm


def build_rainfall_features(
    observations: Iterable[RainfallObservation],
    prediction_time: datetime,
) -> RainfallFeatures:
    """
    Build accumulated rainfall features ending at prediction_time.

    Input readings may arrive out of order; they are sorted internally.
    """

    if prediction_time.tzinfo is None:
        raise ValueError(
            "prediction_time must be timezone-aware."
        )

    ordered = _validate_observations(observations)

    return RainfallFeatures(
        rain_15m_mm=_accumulate_window(
            ordered,
            prediction_time,
            timedelta(minutes=15),
        ),
        rain_30m_mm=_accumulate_window(
            ordered,
            prediction_time,
            timedelta(minutes=30),
        ),
        rain_1h_mm=_accumulate_window(
            ordered,
            prediction_time,
            timedelta(hours=1),
        ),
        rain_3h_mm=_accumulate_window(
            ordered,
            prediction_time,
            timedelta(hours=3),
        ),
        rain_6h_mm=_accumulate_window(
            ordered,
            prediction_time,
            timedelta(hours=6),
        ),
        rain_24h_mm=_accumulate_window(
            ordered,
            prediction_time,
            timedelta(hours=24),
        ),
    )