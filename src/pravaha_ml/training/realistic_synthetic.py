from typing import List

import numpy as np

from pravaha_ml.features.hydrology_features import HydrologyFeatures
from pravaha_ml.training.synthetic import TrainingSample


def _clip(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return float(
        np.clip(
            value,
            minimum,
            maximum,
        )
    )


def _generate_feature_row(
    rng: np.random.Generator,
    latent_risk: float,
) -> HydrologyFeatures:
    """
    Generate development-only synthetic hydrology features.

    Unlike the original simple synthetic generator, distributions
    intentionally overlap so flood and non-flood cases are not
    perfectly separable.

    This is NOT scientifically validated hydrological data.
    """

    rainfall_signal = (
        15.0
        + 90.0 * latent_risk
        + rng.normal(0.0, 22.0)
    )

    rain_1h = _clip(
        rainfall_signal,
        0.0,
        150.0,
    )

    soil_saturation = _clip(
        0.30
        + 0.55 * latent_risk
        + rng.normal(0.0, 0.15),
        0.05,
        1.0,
    )

    base_curve_number = _clip(
        rng.normal(
            72.0 + 8.0 * latent_risk,
            9.0,
        ),
        40.0,
        95.0,
    )

    effective_curve_number = _clip(
        base_curve_number
        + (soil_saturation - 0.50) * 20.0
        + rng.normal(0.0, 4.0),
        35.0,
        100.0,
    )

    runoff_ratio = _clip(
        0.05
        + 0.55 * latent_risk
        + 0.20 * soil_saturation
        + rng.normal(0.0, 0.15),
        0.0,
        0.95,
    )

    runoff_mm = rain_1h * runoff_ratio

    rain_15m = _clip(
        rain_1h
        * rng.uniform(0.15, 0.38)
        + rng.normal(0.0, 3.0),
        0.0,
        100.0,
    )

    rain_30m = _clip(
        rain_1h
        * rng.uniform(0.35, 0.70)
        + rng.normal(0.0, 5.0),
        0.0,
        130.0,
    )

    rain_3h = _clip(
        rain_1h
        * rng.uniform(1.0, 2.5)
        + rng.normal(0.0, 15.0),
        0.0,
        350.0,
    )

    rain_6h = _clip(
        rain_1h
        * rng.uniform(1.3, 3.5)
        + rng.normal(0.0, 20.0),
        0.0,
        500.0,
    )

    rain_24h = _clip(
        rain_1h
        * rng.uniform(1.8, 6.0)
        + rng.normal(0.0, 30.0),
        0.0,
        800.0,
    )

    api_mm = _clip(
        rain_1h
        * rng.uniform(1.0, 4.5)
        + rng.normal(0.0, 20.0),
        0.0,
        600.0,
    )

    flow_length_m = float(
        rng.uniform(
            500.0,
            5000.0,
        )
    )

    slope_fraction = _clip(
        rng.normal(
            0.10 + 0.10 * latent_risk,
            0.07,
        ),
        0.01,
        0.50,
    )

    concentration_time_minutes = _clip(
        85.0
        - 55.0 * latent_risk
        + rng.normal(0.0, 20.0),
        5.0,
        180.0,
    )

    if soil_saturation >= 0.70:
        moisture_condition = "WET"
    elif soil_saturation < 0.35:
        moisture_condition = "DRY"
    else:
        moisture_condition = "NORMAL"

    return HydrologyFeatures(
        rain_15m_mm=rain_15m,
        rain_30m_mm=rain_30m,
        rain_1h_mm=rain_1h,
        rain_3h_mm=rain_3h,
        rain_6h_mm=rain_6h,
        rain_24h_mm=rain_24h,
        api_mm=api_mm,
        soil_moisture_percentage=(
            soil_saturation * 100.0
        ),
        soil_saturation=soil_saturation,
        moisture_condition=moisture_condition,
        base_curve_number=base_curve_number,
        effective_curve_number=effective_curve_number,
        runoff_mm=float(runoff_mm),
        runoff_ratio=runoff_ratio,
        flow_length_m=flow_length_m,
        slope_fraction=slope_fraction,
        concentration_time_minutes=(
            concentration_time_minutes
        ),
    )


def generate_realistic_synthetic_data(
    n_samples: int = 1000,
    flood_fraction: float = 0.30,
    random_state: int = 42,
) -> List[TrainingSample]:
    """
    Generate overlapping development-only flood/non-flood samples.

    Labels are probabilistic rather than directly thresholded from
    any single hydrological variable.
    """

    if n_samples < 10:
        raise ValueError(
            "n_samples must be at least 10."
        )

    if not 0.0 < flood_fraction < 1.0:
        raise ValueError(
            "flood_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(
        random_state
    )

    samples: List[TrainingSample] = []

    for _ in range(n_samples):
        latent_risk = float(
            rng.beta(
                2.0,
                2.0,
            )
        )

        features = _generate_feature_row(
            rng=rng,
            latent_risk=latent_risk,
        )

        # Flood probability depends on combined latent conditions,
        # but includes uncertainty and overlapping classes.
        raw_probability = (
            flood_fraction
            + 0.85 * (latent_risk - 0.50)
            + rng.normal(0.0, 0.10)
        )

        flood_probability = _clip(
            raw_probability,
            0.02,
            0.98,
        )

        label = int(
            rng.random()
            < flood_probability
        )

        samples.append(
            TrainingSample(
                features=features,
                label=label,
            )
        )

    labels = {
        sample.label
        for sample in samples
    }

    if labels != {0, 1}:
        raise RuntimeError(
            "Synthetic generator failed to produce both classes."
        )

    return samples