from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.training.synthetic import (
    TrainingSample,
)


@dataclass(frozen=True)
class HistoricalRecord:
    """
    One labelled historical training observation.

    Metadata fields preserve where and when the observation
    originated.

    HydrologyFeatures contains the actual numeric/model features
    used by PRAVAHA.

    label:
        0 = no flash-flood event
        1 = flash-flood event
    """

    event_id: str
    timestamp: datetime

    latitude: float
    longitude: float

    source: str

    features: HydrologyFeatures

    label: int


REQUIRED_COLUMNS = [
    "event_id",
    "timestamp",
    "latitude",
    "longitude",
    "source",
    "label",

    "rain_15m_mm",
    "rain_30m_mm",
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "api_mm",

    "soil_moisture_percentage",
    "soil_saturation",
    "moisture_condition",

    "base_curve_number",
    "effective_curve_number",

    "runoff_mm",
    "runoff_ratio",

    "flow_length_m",
    "slope_fraction",
    "concentration_time_minutes",
]


def _validate_label(
    label: int,
) -> None:
    if label not in {0, 1}:
        raise ValueError(
            "label must be 0 or 1."
        )


def _validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(
            "latitude must be between -90 and 90."
        )

    if not -180.0 <= longitude <= 180.0:
        raise ValueError(
            "longitude must be between -180 and 180."
        )


def _validate_feature_values(
    features: HydrologyFeatures,
) -> None:
    if not 0.0 <= features.soil_saturation <= 1.0:
        raise ValueError(
            "soil_saturation must be between 0 and 1."
        )

    if not (
        0.0
        <= features.soil_moisture_percentage
        <= 100.0
    ):
        raise ValueError(
            "soil_moisture_percentage must be between 0 and 100."
        )

    if features.base_curve_number <= 0.0:
        raise ValueError(
            "base_curve_number must be greater than 0."
        )

    if features.effective_curve_number <= 0.0:
        raise ValueError(
            "effective_curve_number must be greater than 0."
        )

    if features.runoff_mm < 0.0:
        raise ValueError(
            "runoff_mm cannot be negative."
        )

    if not 0.0 <= features.runoff_ratio <= 1.0:
        raise ValueError(
            "runoff_ratio must be between 0 and 1."
        )

    if features.flow_length_m <= 0.0:
        raise ValueError(
            "flow_length_m must be greater than 0."
        )

    if features.slope_fraction <= 0.0:
        raise ValueError(
            "slope_fraction must be greater than 0."
        )

    if features.concentration_time_minutes <= 0.0:
        raise ValueError(
            "concentration_time_minutes must be greater than 0."
        )


def historical_record_to_training_sample(
    record: HistoricalRecord,
) -> TrainingSample:
    """
    Convert a validated historical record into the common
    TrainingSample interface used by PRAVAHA's training pipelines.
    """

    _validate_label(
        record.label
    )

    _validate_coordinates(
        latitude=record.latitude,
        longitude=record.longitude,
    )

    _validate_feature_values(
        record.features
    )

    return TrainingSample(
        features=record.features,
        label=record.label,
    )


def load_historical_csv(
    path: str | Path,
) -> list[HistoricalRecord]:
    """
    Load a historical labelled feature dataset from CSV.

    This loader expects already-derived hydrology features.

    Raw rainfall, DEM, land-cover and event databases should be
    processed upstream before producing this training table.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Historical dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    records: list[HistoricalRecord] = []

    for row_number, row in dataframe.iterrows():
        try:
            timestamp = pd.to_datetime(
                row["timestamp"],
                utc=True,
            ).to_pydatetime()

            label = int(
                row["label"]
            )

            features = HydrologyFeatures(
                rain_15m_mm=float(
                    row["rain_15m_mm"]
                ),
                rain_30m_mm=float(
                    row["rain_30m_mm"]
                ),
                rain_1h_mm=float(
                    row["rain_1h_mm"]
                ),
                rain_3h_mm=float(
                    row["rain_3h_mm"]
                ),
                rain_6h_mm=float(
                    row["rain_6h_mm"]
                ),
                rain_24h_mm=float(
                    row["rain_24h_mm"]
                ),
                api_mm=float(
                    row["api_mm"]
                ),
                soil_moisture_percentage=float(
                    row[
                        "soil_moisture_percentage"
                    ]
                ),
                soil_saturation=float(
                    row["soil_saturation"]
                ),
                moisture_condition=str(
                    row["moisture_condition"]
                ),
                base_curve_number=float(
                    row["base_curve_number"]
                ),
                effective_curve_number=float(
                    row[
                        "effective_curve_number"
                    ]
                ),
                runoff_mm=float(
                    row["runoff_mm"]
                ),
                runoff_ratio=float(
                    row["runoff_ratio"]
                ),
                flow_length_m=float(
                    row["flow_length_m"]
                ),
                slope_fraction=float(
                    row["slope_fraction"]
                ),
                concentration_time_minutes=float(
                    row[
                        "concentration_time_minutes"
                    ]
                ),
            )

            record = HistoricalRecord(
                event_id=str(
                    row["event_id"]
                ),
                timestamp=timestamp,
                latitude=float(
                    row["latitude"]
                ),
                longitude=float(
                    row["longitude"]
                ),
                source=str(
                    row["source"]
                ),
                features=features,
                label=label,
            )

            _validate_label(
                record.label
            )

            _validate_coordinates(
                latitude=record.latitude,
                longitude=record.longitude,
            )

            _validate_feature_values(
                record.features
            )

            records.append(
                record
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid historical record at row "
                f"{row_number}: {exc}"
            ) from exc

    if not records:
        raise ValueError(
            "Historical dataset contains no records."
        )

    labels = {
        record.label
        for record in records
    }

    if labels != {0, 1}:
        raise ValueError(
            "Historical dataset must contain both "
            "flood and non-flood labels."
        )

    return records


def historical_records_to_training_samples(
    records: Iterable[HistoricalRecord],
) -> list[TrainingSample]:
    """
    Convert historical records into the common model-training
    representation.
    """

    records = list(records)

    if not records:
        raise ValueError(
            "At least one historical record is required."
        )

    samples = [
        historical_record_to_training_sample(
            record
        )
        for record in records
    ]

    labels = {
        sample.label
        for sample in samples
    }

    if labels != {0, 1}:
        raise ValueError(
            "Historical training samples must contain "
            "both classes."
        )

    return samples