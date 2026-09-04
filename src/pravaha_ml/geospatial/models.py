from dataclasses import dataclass
from enum import Enum

from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)


class SpatialDataProvenance(str, Enum):
    VERIFIED = "VERIFIED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(
                "latitude must be between -90 and 90."
            )

        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(
                "longitude must be between -180 and 180."
            )

    def to_shapely(self) -> Point:
        """
        Shapely uses x/y ordering:

            x = longitude
            y = latitude
        """

        return Point(
            self.longitude,
            self.latitude,
        )


@dataclass(frozen=True)
class CatchmentProfile:
    """
    Static GIS/hydrology description of one catchment.

    Geometry is expressed as geographic longitude/latitude
    coordinates.

    base_curve_number:
        Static Curve Number derived from land-cover and soil data.

    flow_length_m:
        Longest hydrological flow path.

    slope_fraction:
        Dimensionless catchment/channel slope.

        Example:
            10% slope = 0.10

    mean_elevation_m:
        Representative catchment elevation above mean sea level.
    """

    catchment_id: str
    polygon: Polygon

    base_curve_number: float
    flow_length_m: float
    slope_fraction: float
    mean_elevation_m: float

    provenance: SpatialDataProvenance

    def __post_init__(self) -> None:
        if not self.catchment_id.strip():
            raise ValueError(
                "catchment_id cannot be empty."
            )

        if self.polygon.is_empty:
            raise ValueError(
                "catchment polygon cannot be empty."
            )

        if not self.polygon.is_valid:
            raise ValueError(
                "catchment polygon must be valid."
            )

        if self.polygon.area <= 0.0:
            raise ValueError(
                "catchment polygon must have positive area."
            )

        if not 1.0 <= self.base_curve_number <= 100.0:
            raise ValueError(
                "base_curve_number must be between 1 and 100."
            )

        if self.flow_length_m <= 0.0:
            raise ValueError(
                "flow_length_m must be greater than 0."
            )

        if self.slope_fraction <= 0.0:
            raise ValueError(
                "slope_fraction must be greater than 0."
            )


@dataclass(frozen=True)
class DrainageSegment:
    """
    Static spatial representation of one drainage segment.

    Hydraulic capacity and dynamic drainage intelligence will be
    added in the drainage-specific layer.
    """

    drain_id: str
    geometry: LineString
    provenance: SpatialDataProvenance

    catchment_id: str | None = None

    def __post_init__(self) -> None:
        if not self.drain_id.strip():
            raise ValueError(
                "drain_id cannot be empty."
            )

        if self.geometry.is_empty:
            raise ValueError(
                "drain geometry cannot be empty."
            )

        if self.geometry.length <= 0.0:
            raise ValueError(
                "drain geometry must have positive length."
            )


@dataclass(frozen=True)
class RoadSegment:
    """
    Static representation of a routable road segment.

    Dynamic flood risk, predicted water depth and routing cost
    belong to later layers.
    """

    road_id: str
    geometry: LineString

    road_name: str | None
    provenance: SpatialDataProvenance

    catchment_id: str | None = None

    def __post_init__(self) -> None:
        if not self.road_id.strip():
            raise ValueError(
                "road_id cannot be empty."
            )

        if self.geometry.is_empty:
            raise ValueError(
                "road geometry cannot be empty."
            )

        if self.geometry.length <= 0.0:
            raise ValueError(
                "road geometry must have positive length."
            )