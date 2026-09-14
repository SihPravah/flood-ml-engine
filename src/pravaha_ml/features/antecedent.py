from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from pravaha_ml.features.rainfall import RainfallObservation


@dataclass(frozen=True)
class AntecedentFeatures:
    """
    Antecedent rainfall state used to represent how wet the catchment
    has been before the current prediction time.
    """

    api_mm: float


def calculate_api(
    observations: Iterable[RainfallObservation],
    prediction_time: datetime,
    decay_factor: float = 0.90,
) -> float:
    """
    Calculate Antecedent Precipitation Index (API).

    API is a simple representation of accumulated catchment wetness.

    Newer rainfall contributes more strongly.
    Older rainfall gradually loses influence through the decay factor.

    Parameters
    ----------
    observations:
        Historical rainfall intensity observations.

    prediction_time:
        Time at which the current state is being estimated.

    decay_factor:
        Value between 0 and 1.

        Higher values mean rainfall remains influential for longer.

        Example:
            0.90 -> relatively slow decay
            0.70 -> faster decay

    Returns
    -------
    float
        Antecedent precipitation index in approximate rainfall-depth units.
    """

    if prediction_time.tzinfo is None:
        raise ValueError(
            "prediction_time must be timezone-aware."
        )

    if not 0.0 <= decay_factor <= 1.0:
        raise ValueError(
            "decay_factor must be between 0 and 1."
        )

    ordered = sorted(
        list(observations),
        key=lambda obs: obs.timestamp,
    )

    if not ordered:
        return 0.0

    for observation in ordered:
        if observation.timestamp.tzinfo is None:
            raise ValueError(
                "Rainfall observation timestamps must be timezone-aware."
            )

        if observation.rainfall_mm_per_hr < 0:
            raise ValueError(
                "Rainfall intensity cannot be negative."
            )

    api = 0.0

    for index, observation in enumerate(ordered):
        if observation.timestamp >= prediction_time:
            continue

        if index + 1 < len(ordered):
            next_time = min(
                ordered[index + 1].timestamp,
                prediction_time,
            )
        else:
            next_time = prediction_time

        if next_time <= observation.timestamp:
            continue

        duration_hours = (
            next_time - observation.timestamp
        ).total_seconds() / 3600.0

        rainfall_depth_mm = (
            observation.rainfall_mm_per_hr
            * duration_hours
        )

        api = rainfall_depth_mm + decay_factor * api

    return api


def build_antecedent_features(
    observations: Iterable[RainfallObservation],
    prediction_time: datetime,
    decay_factor: float = 0.90,
) -> AntecedentFeatures:
    """
    Build antecedent precipitation features.
    """

    return AntecedentFeatures(
        api_mm=calculate_api(
            observations=observations,
            prediction_time=prediction_time,
            decay_factor=decay_factor,
        )
    )