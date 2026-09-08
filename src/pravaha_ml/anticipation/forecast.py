from dataclasses import dataclass
from enum import Enum


class ForecastProvenance(str, Enum):
    """
    Origin of future rainfall information.

    AUTHORITY_FORECAST:
        Forecast supplied by an authoritative meteorological source.

    NWP_DERIVED:
        Derived from numerical weather prediction.

    ENSEMBLE_DERIVED:
        Derived from multiple forecast members.

    ESTIMATED:
        Development/demo estimate rather than authoritative forecast.
    """

    AUTHORITY_FORECAST = "AUTHORITY_FORECAST"
    NWP_DERIVED = "NWP_DERIVED"
    ENSEMBLE_DERIVED = "ENSEMBLE_DERIVED"
    ESTIMATED = "ESTIMATED"


class ForecastScenarioType(str, Enum):
    WEAKENS = "RAIN_WEAKENS"
    CONTINUES = "RAIN_CONTINUES"
    INTENSIFIES = "RAIN_INTENSIFIES"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class ForecastRainfallPoint:
    """
    Cumulative forecast rainfall from now until one future horizon.

    Example:

        horizon_minutes = 60
        cumulative_rainfall_mm = 42

    means 42 mm is expected/assumed during the next 60 minutes.

    This is NOT observed rainfall.
    """

    horizon_minutes: int
    cumulative_rainfall_mm: float

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError(
                "horizon_minutes must be greater than 0."
            )

        if self.cumulative_rainfall_mm < 0.0:
            raise ValueError(
                "cumulative_rainfall_mm cannot be negative."
            )


@dataclass(frozen=True)
class RainfallForecast:
    """
    Baseline short-horizon rainfall forecast.

    Points must:
        - have unique increasing horizons
        - have non-decreasing cumulative rainfall

    probability:
        Probability assigned by the forecast source, when available.

        None means no calibrated probability was provided.

    forecast_confidence:
        Confidence in this forecast input itself.

        This remains separate from hazard-model confidence.
    """

    forecast_id: str

    points: tuple[
        ForecastRainfallPoint,
        ...
    ]

    provenance: ForecastProvenance

    forecast_confidence: float

    probability: float | None = None

    def __post_init__(self) -> None:
        if not self.forecast_id.strip():
            raise ValueError(
                "forecast_id cannot be empty."
            )

        if not self.points:
            raise ValueError(
                "Rainfall forecast requires at least one point."
            )

        if not 0.0 <= self.forecast_confidence <= 1.0:
            raise ValueError(
                "forecast_confidence must be between 0 and 1."
            )

        if (
            self.probability is not None
            and not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError(
                "probability must be between 0 and 1."
            )

        horizons = [
            point.horizon_minutes
            for point in self.points
        ]

        if len(
            horizons
        ) != len(
            set(horizons)
        ):
            raise ValueError(
                "Forecast horizons must be unique."
            )

        if horizons != sorted(
            horizons
        ):
            raise ValueError(
                "Forecast horizons must be increasing."
            )

        cumulative_values = [
            point.cumulative_rainfall_mm
            for point in self.points
        ]

        for index in range(
            1,
            len(cumulative_values),
        ):
            if (
                cumulative_values[index]
                < cumulative_values[index - 1]
            ):
                raise ValueError(
                    "Cumulative forecast rainfall must "
                    "be non-decreasing."
                )


@dataclass(frozen=True)
class ForecastScenario:
    """
    One explicit future rainfall scenario.

    A scenario is a forcing assumption.

    It must not automatically be interpreted as the future that
    will occur.

    Scenario likelihood and downstream hazard consequence are
    deliberately kept separate.
    """

    scenario_id: str

    scenario_type: ForecastScenarioType

    rainfall_points: tuple[
        ForecastRainfallPoint,
        ...
    ]

    provenance: ForecastProvenance

    forecast_confidence: float

    probability: float | None

    rainfall_multiplier: float

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError(
                "scenario_id cannot be empty."
            )

        if not self.rainfall_points:
            raise ValueError(
                "Forecast scenario requires rainfall points."
            )

        if self.rainfall_multiplier < 0.0:
            raise ValueError(
                "rainfall_multiplier cannot be negative."
            )

        if not 0.0 <= self.forecast_confidence <= 1.0:
            raise ValueError(
                "forecast_confidence must be between 0 and 1."
            )

        if (
            self.probability is not None
            and not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError(
                "probability must be between 0 and 1."
            )