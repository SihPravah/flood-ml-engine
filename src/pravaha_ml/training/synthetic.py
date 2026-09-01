from dataclasses import dataclass
from typing import List

import numpy as np

from pravaha_ml.features.hydrology_features import HydrologyFeatures


@dataclass(frozen=True)
class TrainingSample:
    features: HydrologyFeatures
    label: int


def _build_feature_row(
    rng: np.random.Generator,
    is_flood: bool,
) -> HydrologyFeatures:
    """
    Generate deterministic development-only hydrology features.

    IMPORTANT:
    These samples are synthetic and must never be interpreted
    as real hydrological observations or scientific validation.
    """

    if is_flood:
        rain_1h = rng.uniform(55.0, 130.0)
        soil_saturation = rng.uniform(0.70, 0.98)
        effective_cn = rng.uniform(85.0, 98.0)
        runoff_ratio = rng.uniform(0.40, 0.80)
        slope_fraction = rng.uniform(0.08, 0.35)
        concentration_time = rng.uniform(10.0, 45.0)

    else:
        rain_1h = rng.uniform(0.0, 50.0)
        soil_saturation = rng.uniform(0.20, 0.75)
        effective_cn = rng.uniform(55.0, 88.0)
        runoff_ratio = rng.uniform(0.0, 0.40)
        slope_fraction = rng.uniform(0.02, 0.25)
        concentration_time = rng.uniform(25.0, 120.0)

    soil_moisture_percentage = soil_saturation * 100.0

    runoff_mm = rain_1h * runoff_ratio

    rain_15m = rain_1h * rng.uniform(0.18, 0.32)
    rain_30m = rain_1h * rng.uniform(0.40, 0.60)

    rain_3h = rain_1h * rng.uniform(1.2, 2.4)
    rain_6h = rain_1h * rng.uniform(1.8, 3.5)
    rain_24h = rain_1h * rng.uniform(2.5, 6.0)

    api_mm = rain_1h * rng.uniform(1.5, 4.5)

    base_curve_number = np.clip(
        effective_cn - rng.uniform(-3.0, 6.0),
        40.0,
        95.0,
    )

    flow_length_m = rng.uniform(
        500.0,
        5000.0,
    )

    if soil_saturation >= 0.70:
        moisture_condition = "WET"
    elif soil_saturation < 0.35:
        moisture_condition = "DRY"
    else:
        moisture_condition = "NORMAL"

    return HydrologyFeatures(
        rain_15m_mm=float(rain_15m),
        rain_30m_mm=float(rain_30m),
        rain_1h_mm=float(rain_1h),
        rain_3h_mm=float(rain_3h),
        rain_6h_mm=float(rain_6h),
        rain_24h_mm=float(rain_24h),
        api_mm=float(api_mm),
        soil_moisture_percentage=float(
            soil_moisture_percentage
        ),
        soil_saturation=float(soil_saturation),
        moisture_condition=moisture_condition,
        base_curve_number=float(base_curve_number),
        effective_curve_number=float(effective_cn),
        runoff_mm=float(runoff_mm),
        runoff_ratio=float(runoff_ratio),
        flow_length_m=float(flow_length_m),
        slope_fraction=float(slope_fraction),
        concentration_time_minutes=float(
            concentration_time
        ),
    )


def generate_synthetic_training_data(
    n_samples: int = 500,
    flood_fraction: float = 0.35,
    random_state: int = 42,
) -> List[TrainingSample]:
    if n_samples < 2:
        raise ValueError(
            "n_samples must be at least 2."
        )

    if not 0.0 < flood_fraction < 1.0:
        raise ValueError(
            "flood_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(
        random_state
    )

    n_flood = max(
        1,
        int(round(n_samples * flood_fraction)),
    )

    n_non_flood = n_samples - n_flood

    if n_non_flood == 0:
        n_non_flood = 1
        n_flood = n_samples - 1

    samples: List[TrainingSample] = []

    for _ in range(n_non_flood):
        samples.append(
            TrainingSample(
                features=_build_feature_row(
                    rng=rng,
                    is_flood=False,
                ),
                label=0,
            )
        )

    for _ in range(n_flood):
        samples.append(
            TrainingSample(
                features=_build_feature_row(
                    rng=rng,
                    is_flood=True,
                ),
                label=1,
            )
        )

    order = rng.permutation(
        len(samples)
    )

    return [
        samples[index]
        for index in order
    ]