from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

from pravaha_ml.data.historical import (
    HistoricalRecord,
    historical_record_to_training_sample,
)
from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
    build_hydrology_features,
)
from pravaha_ml.features.rainfall import (
    RainfallObservation,
)
from pravaha_ml.training.synthetic import (
    TrainingSample,
)


class SoilMoistureProvenance(str, Enum):
    OBSERVED = "OBSERVED"
    ESTIMATED = "ESTIMATED"


@dataclass(frozen=True)
class HistoricalStaticCatchmentData:
    """
    Static catchment properties used by the hydrology engine.

    These values are expected to come from future GIS/DEM/
    land-cover preprocessing.

    They are not part of the live sensor contract.
    """

    base_curve_number: float
    flow_length_m: float
    slope_fraction: float


@dataclass(frozen=True)
class HistoricalSoilMoistureInput:
    """
    Historical soil-moisture information.

    observed_percentage:
        Actual historical observation when available.

    estimated_percentage:
        Explicit externally-derived estimate when an observation
        is unavailable.

    PRAVAHA does not silently invent a soil-moisture value.

    If neither value is available, feature construction fails.

    When both are supplied, the real observation takes priority.
    """

    observed_percentage: float | None = None
    estimated_percentage: float | None = None


@dataclass(frozen=True)
class ResolvedSoilMoisture:
    value_percentage: float
    provenance: SoilMoistureProvenance
    was_missing_observation: bool


@dataclass(frozen=True)
class HistoricalFeatureProvenance:
    """
    Provenance attached to a derived historical feature row.

    This allows downstream training/evaluation code to know
    whether soil moisture was measured or estimated.
    """

    soil_moisture_provenance: SoilMoistureProvenance
    soil_moisture_was_missing: bool

    rainfall_observation_count: int

    hydrology_feature_engine: str


@dataclass(frozen=True)
class HistoricalFeatureBuildResult:
    record: HistoricalRecord
    training_sample: TrainingSample
    provenance: HistoricalFeatureProvenance


def _validate_percentage(
    value: float,
    field_name: str,
) -> None:
    if not 0.0 <= value <= 100.0:
        raise ValueError(
            f"{field_name} must be between 0 and 100."
        )


def resolve_soil_moisture(
    soil_moisture: HistoricalSoilMoistureInput,
) -> ResolvedSoilMoisture:
    """
    Resolve historical soil moisture without silently fabricating
    a value.

    Priority:

        observed value
            ↓
        estimated value
            ↓
        fail explicitly

    An estimated value must therefore come from an explicit
    upstream estimation/data-fusion process.
    """

    observed = (
        soil_moisture.observed_percentage
    )

    estimated = (
        soil_moisture.estimated_percentage
    )

    if observed is not None:
        _validate_percentage(
            value=observed,
            field_name=(
                "observed soil moisture"
            ),
        )

        return ResolvedSoilMoisture(
            value_percentage=float(
                observed
            ),
            provenance=(
                SoilMoistureProvenance.OBSERVED
            ),
            was_missing_observation=False,
        )

    if estimated is not None:
        _validate_percentage(
            value=estimated,
            field_name=(
                "estimated soil moisture"
            ),
        )

        return ResolvedSoilMoisture(
            value_percentage=float(
                estimated
            ),
            provenance=(
                SoilMoistureProvenance.ESTIMATED
            ),
            was_missing_observation=True,
        )

    raise ValueError(
        "Historical soil moisture is unavailable. "
        "Provide either an observed value or an explicit "
        "estimated value with provenance."
    )


def _validate_static_catchment_data(
    data: HistoricalStaticCatchmentData,
) -> None:
    if not 1.0 <= data.base_curve_number <= 100.0:
        raise ValueError(
            "base_curve_number must be between 1 and 100."
        )

    if data.flow_length_m <= 0.0:
        raise ValueError(
            "flow_length_m must be greater than 0."
        )

    if data.slope_fraction <= 0.0:
        raise ValueError(
            "slope_fraction must be greater than 0."
        )


def build_historical_feature_record(
    *,
    event_id: str,
    prediction_time: datetime,
    latitude: float,
    longitude: float,
    source: str,
    label: int,
    rainfall_observations: Iterable[
        RainfallObservation
    ],
    soil_moisture: HistoricalSoilMoistureInput,
    catchment: HistoricalStaticCatchmentData,
    dry_threshold_percentage: float,
    wet_threshold_percentage: float,
    api_decay_factor: float = 0.90,
) -> HistoricalFeatureBuildResult:
    """
    Convert raw historical environmental inputs into the same
    HydrologyFeatures representation used by PRAVAHA's ML layer.

    Pipeline:

        raw rainfall history
                +
        observed/estimated soil moisture
                +
        static catchment properties
                ↓
        shared hydrology feature engine
                ↓
        HistoricalRecord
                ↓
        TrainingSample

    This prevents training-serving skew because historical
    training features and future live-inference features use
    the same hydrological calculations.
    """

    rainfall_observations = list(
        rainfall_observations
    )

    if not rainfall_observations:
        raise ValueError(
            "At least one historical rainfall observation "
            "is required."
        )

    if label not in {0, 1}:
        raise ValueError(
            "label must be 0 or 1."
        )

    if not event_id.strip():
        raise ValueError(
            "event_id cannot be empty."
        )

    if not source.strip():
        raise ValueError(
            "source cannot be empty."
        )

    if not -90.0 <= latitude <= 90.0:
        raise ValueError(
            "latitude must be between -90 and 90."
        )

    if not -180.0 <= longitude <= 180.0:
        raise ValueError(
            "longitude must be between -180 and 180."
        )

    _validate_static_catchment_data(
        catchment
    )

    resolved_soil = resolve_soil_moisture(
        soil_moisture
    )

    features: HydrologyFeatures = (
        build_hydrology_features(
            rainfall_observations=(
                rainfall_observations
            ),
            prediction_time=prediction_time,
            soil_moisture_percentage=(
                resolved_soil.value_percentage
            ),
            base_curve_number=(
                catchment.base_curve_number
            ),
            flow_length_m=(
                catchment.flow_length_m
            ),
            slope_fraction=(
                catchment.slope_fraction
            ),
            dry_threshold_percentage=(
                dry_threshold_percentage
            ),
            wet_threshold_percentage=(
                wet_threshold_percentage
            ),
            api_decay_factor=(
                api_decay_factor
            ),
        )
    )

    record = HistoricalRecord(
        event_id=event_id,
        timestamp=prediction_time,
        latitude=latitude,
        longitude=longitude,
        source=source,
        features=features,
        label=label,
    )

    training_sample = (
        historical_record_to_training_sample(
            record
        )
    )

    provenance = (
        HistoricalFeatureProvenance(
            soil_moisture_provenance=(
                resolved_soil.provenance
            ),
            soil_moisture_was_missing=(
                resolved_soil
                .was_missing_observation
            ),
            rainfall_observation_count=len(
                rainfall_observations
            ),
            hydrology_feature_engine=(
                "pravaha-hydrology-v1"
            ),
        )
    )

    return HistoricalFeatureBuildResult(
        record=record,
        training_sample=training_sample,
        provenance=provenance,
    )