import pytest

from shapely.geometry import Polygon

from pravaha_ml.geospatial.catchments import (
    AmbiguousCatchmentError,
    CatchmentNotFoundError,
    find_catchment_by_id,
    find_catchment_for_point,
)
from pravaha_ml.geospatial.models import (
    CatchmentProfile,
    GeoPoint,
    SpatialDataProvenance,
)


def make_catchment(
    catchment_id: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> CatchmentProfile:
    polygon = Polygon(
        [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
    )

    return CatchmentProfile(
        catchment_id=catchment_id,
        polygon=polygon,
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        mean_elevation_m=1200.0,
        provenance=(
            SpatialDataProvenance.DERIVED
        ),
    )


def test_point_maps_to_correct_catchment():
    catchment_a = make_catchment(
        "C_A",
        77.00,
        30.00,
        77.10,
        30.10,
    )

    catchment_b = make_catchment(
        "C_B",
        77.20,
        30.20,
        77.30,
        30.30,
    )

    point = GeoPoint(
        latitude=30.05,
        longitude=77.05,
    )

    result = find_catchment_for_point(
        point=point,
        catchments=[
            catchment_a,
            catchment_b,
        ],
    )

    assert (
        result.catchment.catchment_id
        == "C_A"
    )

    assert (
        result.matched_by
        == "polygon_cover"
    )


def test_boundary_point_is_accepted():
    catchment = make_catchment(
        "C_A",
        77.00,
        30.00,
        77.10,
        30.10,
    )

    point = GeoPoint(
        latitude=30.05,
        longitude=77.00,
    )

    result = find_catchment_for_point(
        point=point,
        catchments=[
            catchment
        ],
    )

    assert (
        result.catchment.catchment_id
        == "C_A"
    )


def test_point_outside_all_catchments_rejected():
    catchment = make_catchment(
        "C_A",
        77.00,
        30.00,
        77.10,
        30.10,
    )

    point = GeoPoint(
        latitude=31.00,
        longitude=78.00,
    )

    with pytest.raises(
        CatchmentNotFoundError,
        match=(
            "No catchment contains"
        ),
    ):
        find_catchment_for_point(
            point=point,
            catchments=[
                catchment
            ],
        )


def test_overlapping_catchments_are_not_silently_resolved():
    catchment_a = make_catchment(
        "C_A",
        77.00,
        30.00,
        77.10,
        30.10,
    )

    catchment_b = make_catchment(
        "C_B",
        77.05,
        30.05,
        77.15,
        30.15,
    )

    point = GeoPoint(
        latitude=30.075,
        longitude=77.075,
    )

    with pytest.raises(
        AmbiguousCatchmentError,
        match=(
            "multiple catchments"
        ),
    ):
        find_catchment_for_point(
            point=point,
            catchments=[
                catchment_a,
                catchment_b,
            ],
        )


def test_find_catchment_by_id():
    catchment_a = make_catchment(
        "C_A",
        77.00,
        30.00,
        77.10,
        30.10,
    )

    catchment_b = make_catchment(
        "C_B",
        77.20,
        30.20,
        77.30,
        30.30,
    )

    result = find_catchment_by_id(
        catchment_id="C_B",
        catchments=[
            catchment_a,
            catchment_b,
        ],
    )

    assert (
        result.catchment_id
        == "C_B"
    )


def test_unknown_catchment_id_rejected():
    catchment = make_catchment(
        "C_A",
        77.00,
        30.00,
        77.10,
        30.10,
    )

    with pytest.raises(
        CatchmentNotFoundError,
        match="Catchment not found",
    ):
        find_catchment_by_id(
            catchment_id="UNKNOWN",
            catchments=[
                catchment
            ],
        )


def test_empty_catchment_collection_rejected():
    point = GeoPoint(
        latitude=30.05,
        longitude=77.05,
    )

    with pytest.raises(
        ValueError,
        match=(
            "At least one catchment is required"
        ),
    ):
        find_catchment_for_point(
            point=point,
            catchments=[],
        )