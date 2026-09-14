from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
    TemporalQualityReport,
)
from pravaha_ml.inference.confidence import (
    InputProvenance,
)


_MISSING = object()


class DataStatus(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    SIMULATED = "SIMULATED"
    MISSING = "MISSING"


class FusedWindowQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"


@dataclass(frozen=True)
class FusedIntensity:
    value: float | None
    status: DataStatus
    confidence: float | None
    age_minutes: float | None


@dataclass(frozen=True)
class FusedRainWindow:
    value_mm: float | None
    status: DataStatus
    coverage_fraction: float
    largest_gap_minutes: float | None
    latest_observation_age_minutes: float | None
    observation_count: int
    quality: FusedWindowQuality


@dataclass(frozen=True)
class FusedRainfall:
    intensity: FusedIntensity
    rain_15m: FusedRainWindow
    rain_30m: FusedRainWindow
    rain_1h: FusedRainWindow
    rain_3h: FusedRainWindow
    rain_6h: FusedRainWindow
    rain_24h: FusedRainWindow


@dataclass(frozen=True)
class FusedSoil:
    saturation: float | None
    status: DataStatus
    confidence: float | None
    age_minutes: float | None


@dataclass(frozen=True)
class FusedDataQuality:
    overall_score: float
    missing_sources: tuple[str, ...]
    temporal_freshness: FusedWindowQuality


@dataclass(frozen=True)
class FusedCatchmentStateV21:
    catchment_id: str
    state_time: datetime
    rainfall: FusedRainfall
    soil: FusedSoil
    data_quality: FusedDataQuality

    @property
    def rainfall_windows(self) -> tuple[FusedRainWindow, ...]:
        return (
            self.rainfall.rain_15m,
            self.rainfall.rain_30m,
            self.rainfall.rain_1h,
            self.rainfall.rain_3h,
            self.rainfall.rain_6h,
            self.rainfall.rain_24h,
        )

    def temporal_quality_report(self) -> TemporalQualityReport:
        critical = self.rainfall.rain_1h
        quality_order = {
            FusedWindowQuality.GOOD: 0,
            FusedWindowQuality.DEGRADED: 1,
            FusedWindowQuality.UNUSABLE: 2,
        }

        worst_quality = max(
            (
                critical.quality,
                self.data_quality.temporal_freshness,
            ),
            key=lambda item: quality_order[item],
        )

        reasons: list[str] = []
        if critical.status == DataStatus.MISSING:
            reasons.append("missing_rain_1h")
        if self.soil.status == DataStatus.MISSING:
            reasons.append("missing_soil_saturation")
        if worst_quality == FusedWindowQuality.DEGRADED:
            reasons.append("temporal_data_degraded")
        if worst_quality == FusedWindowQuality.UNUSABLE:
            reasons.append("temporal_data_unusable")

        level = TemporalQualityLevel(worst_quality.value)
        can_predict = (
            level != TemporalQualityLevel.UNUSABLE
            and critical.status != DataStatus.MISSING
            and critical.value_mm is not None
            and self.soil.status != DataStatus.MISSING
            and self.soil.saturation is not None
        )

        return TemporalQualityReport(
            level=level,
            observation_count=critical.observation_count,
            latest_observation_age_minutes=(
                critical.latest_observation_age_minutes
                if critical.latest_observation_age_minutes is not None
                else self.rainfall.intensity.age_minutes
                if self.rainfall.intensity.age_minutes is not None
                else 0.0
            ),
            largest_gap_minutes=(
                critical.largest_gap_minutes
                if critical.largest_gap_minutes is not None
                else 0.0
            ),
            has_future_observation=False,
            has_duplicate_timestamp=False,
            stale=False,
            excessive_gap=(
                critical.quality == FusedWindowQuality.UNUSABLE
                and (
                    critical.largest_gap_minutes is not None
                    and critical.largest_gap_minutes > 30.0
                )
            ),
            can_predict=can_predict,
            reasons=tuple(reasons),
        )

    def source_availability_fraction(self) -> float:
        required = [
            self.rainfall.intensity.status,
            self.rainfall.rain_15m.status,
            self.rainfall.rain_30m.status,
            self.rainfall.rain_1h.status,
            self.rainfall.rain_3h.status,
            self.rainfall.rain_6h.status,
            self.rainfall.rain_24h.status,
            self.soil.status,
        ]
        available = sum(
            status != DataStatus.MISSING
            for status in required
        )
        return available / len(required)

    def confidence_provenances(self) -> tuple[InputProvenance, ...]:
        return tuple(
            _to_confidence_provenance(status)
            for status in (
                self.rainfall.intensity.status,
                self.rainfall.rain_15m.status,
                self.rainfall.rain_30m.status,
                self.rainfall.rain_1h.status,
                self.rainfall.rain_3h.status,
                self.rainfall.rain_6h.status,
                self.rainfall.rain_24h.status,
                self.soil.status,
            )
        )

    def original_provenance(self) -> dict[str, str]:
        return {
            "rainfall.intensity": self.rainfall.intensity.status.value,
            "rainfall.rain_15m": self.rainfall.rain_15m.status.value,
            "rainfall.rain_30m": self.rainfall.rain_30m.status.value,
            "rainfall.rain_1h": self.rainfall.rain_1h.status.value,
            "rainfall.rain_3h": self.rainfall.rain_3h.status.value,
            "rainfall.rain_6h": self.rainfall.rain_6h.status.value,
            "rainfall.rain_24h": self.rainfall.rain_24h.status.value,
            "soil.saturation": self.soil.status.value,
        }


def adapt_fused_catchment_state(
    payload: Mapping[str, Any] | Any,
) -> FusedCatchmentStateV21:
    catchment_id = _as_str(
        _field(payload, "catchment_id"),
        "catchment_id",
    )

    rainfall = _field(payload, "rainfall")
    soil = _field(payload, "soil")
    data_quality = _field(payload, "data_quality")

    return FusedCatchmentStateV21(
        catchment_id=catchment_id,
        state_time=_as_datetime(
            _field(payload, "state_time"),
            "state_time",
        ),
        rainfall=FusedRainfall(
            intensity=_parse_intensity(
                _field(rainfall, "intensity"),
                "rainfall.intensity",
            ),
            rain_15m=_parse_window(
                _field(rainfall, "rain_15m"),
                "rainfall.rain_15m",
            ),
            rain_30m=_parse_window(
                _field(rainfall, "rain_30m"),
                "rainfall.rain_30m",
            ),
            rain_1h=_parse_window(
                _field(rainfall, "rain_1h"),
                "rainfall.rain_1h",
            ),
            rain_3h=_parse_window(
                _field(rainfall, "rain_3h"),
                "rainfall.rain_3h",
            ),
            rain_6h=_parse_window(
                _field(rainfall, "rain_6h"),
                "rainfall.rain_6h",
            ),
            rain_24h=_parse_window(
                _field(rainfall, "rain_24h"),
                "rainfall.rain_24h",
            ),
        ),
        soil=_parse_soil(soil),
        data_quality=FusedDataQuality(
            overall_score=_fraction(
                _field(data_quality, "overall_score"),
                "data_quality.overall_score",
            ),
            missing_sources=tuple(
                str(item)
                for item in (
                    _field(
                        data_quality,
                        "missing_sources",
                        default=[],
                    )
                    or []
                )
            ),
            temporal_freshness=FusedWindowQuality(
                _as_str(
                    _field(
                        data_quality,
                        "temporal_freshness",
                    ),
                    "data_quality.temporal_freshness",
                )
            ),
        ),
    )


def _to_confidence_provenance(
    status: DataStatus,
) -> InputProvenance:
    if status == DataStatus.OBSERVED:
        return InputProvenance.OBSERVED
    if status == DataStatus.MISSING:
        return InputProvenance.MISSING
    return InputProvenance.ESTIMATED


def _parse_intensity(
    value: Mapping[str, Any] | Any,
    path: str,
) -> FusedIntensity:
    return FusedIntensity(
        value=_optional_non_negative(
            _field(value, "value"),
            f"{path}.value",
        ),
        status=DataStatus(
            _as_str(
                _field(value, "status"),
                f"{path}.status",
            )
        ),
        confidence=_optional_fraction(
            _field(value, "confidence", default=None),
            f"{path}.confidence",
        ),
        age_minutes=_optional_non_negative(
            _field(value, "age_minutes", default=None),
            f"{path}.age_minutes",
        ),
    )


def _parse_window(
    value: Mapping[str, Any] | Any,
    path: str,
) -> FusedRainWindow:
    observation_count = int(
        _field(value, "observation_count", default=0)
    )
    if observation_count < 0:
        raise ValueError(f"{path}.observation_count cannot be negative.")

    return FusedRainWindow(
        value_mm=_optional_non_negative(
            _field(value, "value_mm"),
            f"{path}.value_mm",
        ),
        status=DataStatus(
            _as_str(
                _field(value, "status"),
                f"{path}.status",
            )
        ),
        coverage_fraction=_fraction(
            _field(value, "coverage_fraction", default=0.0),
            f"{path}.coverage_fraction",
        ),
        largest_gap_minutes=_optional_non_negative(
            _field(value, "largest_gap_minutes", default=None),
            f"{path}.largest_gap_minutes",
        ),
        latest_observation_age_minutes=_optional_non_negative(
            _field(
                value,
                "latest_observation_age_minutes",
                default=None,
            ),
            f"{path}.latest_observation_age_minutes",
        ),
        observation_count=observation_count,
        quality=FusedWindowQuality(
            _as_str(
                _field(value, "quality"),
                f"{path}.quality",
            )
        ),
    )


def _parse_soil(
    value: Mapping[str, Any] | Any,
) -> FusedSoil:
    saturation = _field(value, "saturation")
    parsed_saturation = _optional_fraction(
        saturation,
        "soil.saturation",
    )
    return FusedSoil(
        saturation=parsed_saturation,
        status=DataStatus(
            _as_str(
                _field(value, "status"),
                "soil.status",
            )
        ),
        confidence=_optional_fraction(
            _field(value, "confidence", default=None),
            "soil.confidence",
        ),
        age_minutes=_optional_non_negative(
            _field(value, "age_minutes", default=None),
            "soil.age_minutes",
        ),
    )


def _field(
    source: Mapping[str, Any] | Any,
    name: str,
    *,
    default: Any = _MISSING,
) -> Any:
    if isinstance(source, Mapping):
        if name in source:
            return source[name]
        if default is not _MISSING:
            return default
        raise ValueError(f"Missing required field: {name}")

    if hasattr(source, name):
        return getattr(source, name)

    if default is not _MISSING:
        return default

    raise ValueError(f"Missing required field: {name}")


def _as_str(
    value: Any,
    path: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _as_datetime(
    value: Any,
    path: str,
) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    else:
        raise ValueError(f"{path} must be a datetime or ISO string.")

    if parsed.tzinfo is None:
        raise ValueError(f"{path} must be timezone-aware.")

    return parsed.astimezone(timezone.utc)


def _optional_non_negative(
    value: Any,
    path: str,
) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"{path} cannot be negative.")
    return parsed


def _optional_fraction(
    value: Any,
    path: str,
) -> float | None:
    if value is None:
        return None
    return _fraction(value, path)


def _fraction(
    value: Any,
    path: str,
) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1.")
    return parsed
