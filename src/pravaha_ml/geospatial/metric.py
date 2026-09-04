from dataclasses import dataclass

from pyproj import CRS, Transformer
from shapely.geometry import (
    LineString,
    Point,
)
from shapely.ops import transform


@dataclass(frozen=True)
class MetricDistanceResult:
    distance_m: float
    projected_crs_epsg: int


def utm_epsg_for_lon_lat(
    *,
    longitude: float,
    latitude: float,
) -> int:
    """
    Select the local WGS84 UTM CRS for a longitude/latitude point.

    Northern hemisphere:
        EPSG 326xx

    Southern hemisphere:
        EPSG 327xx

    Example:
        longitude ~77E in northern India
            -> UTM zone 43N
            -> EPSG:32643

    This helper is intended for ordinary non-polar project areas.
    """

    if not -180.0 <= longitude <= 180.0:
        raise ValueError(
            "longitude must be between -180 and 180."
        )

    if not -80.0 <= latitude <= 84.0:
        raise ValueError(
            "UTM helper supports latitudes between -80 and 84."
        )

    zone = int(
        (longitude + 180.0)
        // 6.0
    ) + 1

    zone = max(
        1,
        min(
            zone,
            60,
        ),
    )

    if latitude >= 0.0:
        return (
            32600
            + zone
        )

    return (
        32700
        + zone
    )


def _combined_reference_point(
    first_geometry: LineString,
    second_geometry: LineString,
) -> Point:
    """
    Construct a representative geographic location for choosing
    the local UTM zone.

    This does not calculate distance itself.
    """

    first_centroid = (
        first_geometry.centroid
    )

    second_centroid = (
        second_geometry.centroid
    )

    longitude = (
        first_centroid.x
        + second_centroid.x
    ) / 2.0

    latitude = (
        first_centroid.y
        + second_centroid.y
    ) / 2.0

    return Point(
        longitude,
        latitude,
    )


def line_distance_metres(
    first_geometry: LineString,
    second_geometry: LineString,
) -> MetricDistanceResult:
    """
    Calculate minimum metric distance between two geographic
    LineString geometries.

    Input geometry coordinates are assumed to be:

        x = longitude
        y = latitude

    The geometries are projected from WGS84 into an appropriate
    local UTM CRS before Shapely distance is calculated.

    The result can therefore legitimately be interpreted as
    metres, unlike raw Shapely distance in EPSG:4326.
    """

    if first_geometry.is_empty:
        raise ValueError(
            "first_geometry cannot be empty."
        )

    if second_geometry.is_empty:
        raise ValueError(
            "second_geometry cannot be empty."
        )

    reference = (
        _combined_reference_point(
            first_geometry,
            second_geometry,
        )
    )

    epsg = utm_epsg_for_lon_lat(
        longitude=reference.x,
        latitude=reference.y,
    )

    source_crs = CRS.from_epsg(
        4326
    )

    target_crs = CRS.from_epsg(
        epsg
    )

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    projected_first = transform(
        transformer.transform,
        first_geometry,
    )

    projected_second = transform(
        transformer.transform,
        second_geometry,
    )

    distance_m = projected_first.distance(
        projected_second
    )

    return MetricDistanceResult(
        distance_m=float(
            distance_m
        ),
        projected_crs_epsg=epsg,
    )