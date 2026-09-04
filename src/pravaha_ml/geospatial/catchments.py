from dataclasses import dataclass
from typing import Iterable

from shapely.geometry import Point

from pravaha_ml.geospatial.models import (
    CatchmentProfile,
    GeoPoint,
)


@dataclass(frozen=True)
class CatchmentMatch:
    catchment: CatchmentProfile
    point: GeoPoint

    matched_by: str


class CatchmentNotFoundError(
    LookupError
):
    """
    Raised when a geographic point cannot be associated with any
    known catchment.
    """


class AmbiguousCatchmentError(
    LookupError
):
    """
    Raised when the same point appears inside more than one
    catchment polygon.

    Overlapping catchment geometries should normally be fixed in
    GIS preprocessing rather than silently choosing one.
    """


def find_catchment_for_point(
    point: GeoPoint,
    catchments: Iterable[
        CatchmentProfile
    ],
) -> CatchmentMatch:
    """
    Find the catchment containing a geographic point.

    Boundary points are accepted using covers() rather than
    contains().

    This means a sensor lying exactly on a polygon boundary can
    still be associated with the catchment.
    """

    catchments = list(
        catchments
    )

    if not catchments:
        raise ValueError(
            "At least one catchment is required."
        )

    shapely_point: Point = (
        point.to_shapely()
    )

    matches = [
        catchment
        for catchment in catchments
        if catchment.polygon.covers(
            shapely_point
        )
    ]

    if not matches:
        raise CatchmentNotFoundError(
            "No catchment contains the supplied point."
        )

    if len(matches) > 1:
        raise AmbiguousCatchmentError(
            "Point matches multiple catchments."
        )

    return CatchmentMatch(
        catchment=matches[0],
        point=point,
        matched_by="polygon_cover",
    )


def find_catchment_by_id(
    catchment_id: str,
    catchments: Iterable[
        CatchmentProfile
    ],
) -> CatchmentProfile:
    catchments = list(
        catchments
    )

    matches = [
        catchment
        for catchment in catchments
        if catchment.catchment_id
        == catchment_id
    ]

    if not matches:
        raise CatchmentNotFoundError(
            f"Catchment not found: {catchment_id}"
        )

    if len(matches) > 1:
        raise AmbiguousCatchmentError(
            f"Duplicate catchment_id: {catchment_id}"
        )

    return matches[0]