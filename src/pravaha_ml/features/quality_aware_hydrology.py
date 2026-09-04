from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterable

from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
    build_hydrology_features,
)
from pravaha_ml.features.rainfall import (
    RainfallObservation,
)
from pravaha_ml.features.temporal_quality import (
    TemporalQualityPolicy,
    TemporalQualityReport,
    calculate_time_aware_api,
    evaluate_temporal_quality,
)


@dataclass(frozen=True)
class QualityAwareHydrologyResult:
    """
    Hydrology features accompanied by temporal input-quality
    metadata.

    A successful result means the temporal quality gate allowed
    feature construction.
    """

    features: HydrologyFeatures
    temporal_quality: TemporalQualityReport


class TemporalDataQualityError(
    ValueError
):
    """
    Raised when temporal input quality is insufficient for a
    reliable hydrology calculation.
    """


def build_quality_aware_hydrology_features(
    *,
    rainfall_observations: Iterable[
        RainfallObservation
    ],
    prediction_time: datetime,
    soil_moisture_percentage: float,
    base_curve_number: float,
    flow_length_m: float,
    slope_fraction: float,
    dry_threshold_percentage: float,
    wet_threshold_percentage: float,
    temporal_quality_policy: (
        TemporalQualityPolicy | None
    ) = None,
    api_decay_factor_per_hour: float = 0.90,
) -> QualityAwareHydrologyResult:
    """
    Build PRAVAHA hydrology features only after temporal-data
    quality has been evaluated.

    Safety behaviour:

        GOOD
            -> prediction features allowed

        DEGRADED
            -> prediction features allowed, but downstream
               confidence should later be reduced

        UNUSABLE
            -> feature construction is blocked

    The time-aware antecedent precipitation value replaces the
    older observation-count-based API value in the resulting
    HydrologyFeatures object.
    """

    rainfall_observations = list(
        rainfall_observations
    )

    quality = evaluate_temporal_quality(
        observations=rainfall_observations,
        prediction_time=prediction_time,
        policy=temporal_quality_policy,
    )

    if not quality.can_predict:
        reasons = (
            ", ".join(
                quality.reasons
            )
            if quality.reasons
            else "unknown temporal quality failure"
        )

        raise TemporalDataQualityError(
            "Temporal rainfall data is not reliable "
            "enough for hydrology feature generation: "
            f"{reasons}"
        )

    base_features = (
        build_hydrology_features(
            rainfall_observations=(
                rainfall_observations
            ),
            prediction_time=prediction_time,
            soil_moisture_percentage=(
                soil_moisture_percentage
            ),
            base_curve_number=(
                base_curve_number
            ),
            flow_length_m=(
                flow_length_m
            ),
            slope_fraction=(
                slope_fraction
            ),
            dry_threshold_percentage=(
                dry_threshold_percentage
            ),
            wet_threshold_percentage=(
                wet_threshold_percentage
            ),
            api_decay_factor=0.90,
        )
    )

    time_aware_api_mm = (
        calculate_time_aware_api(
            observations=(
                rainfall_observations
            ),
            prediction_time=(
                prediction_time
            ),
            decay_factor_per_hour=(
                api_decay_factor_per_hour
            ),
        )
    )

    features = replace(
        base_features,
        api_mm=time_aware_api_mm,
    )

    return QualityAwareHydrologyResult(
        features=features,
        temporal_quality=quality,
    )