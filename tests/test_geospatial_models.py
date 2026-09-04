import pytest

from shapely.geometry import (
    LineString,
    Polygon,
)

from pravaha_ml.geospatial.models import (
    CatchmentProfile,
    DrainageSegment,
    GeoPoint,
    RoadSegment,
    SpatialDataProvenance,
)


def make_polygon() -> Polygon:
    return Polygon(
        [
            (77.00, 30.00),
            (77.10, 30.00),
            (77.10, 30.10),
            (77.00, 30.10),
            (77.00, 30.00),
        ]
    )


def test_geo_point_valid():
    point = GeoPoint(
        latitude=30.05,
        longitude=77.05,
    )

    assert point.latitude == 30.05
    assert point.longitude == 77.05


def test_invalid_latitude_rejected():
    with pytest.raises(
        ValueError,
        match="latitude must be between -90 and 90",
    ):
        GeoPoint(
            latitude=100.0,
            longitude=77.0,
        )


def test_invalid_longitude_rejected():
    with pytest.raises(
        ValueError,
        match="longitude must be between -180 and 180",
    ):
        GeoPoint(
            latitude=30.0,
            longitude=200.0,
        )


def test_geo_point_uses_longitude_as_x():
    point = GeoPoint(
        latitude=30.05,
        longitude=77.05,
    )

    shapely_point = (
        point.to_shapely()
    )

    assert shapely_point.x == pytest.approx(
        77.05
    )

    assert shapely_point.y == pytest.approx(
        30.05
    )


def test_valid_catchment():
    catchment = CatchmentProfile(
        catchment_id="C_001",
        polygon=make_polygon(),
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        mean_elevation_m=1200.0,
        provenance=(
            SpatialDataProvenance.DERIVED
        ),
    )

    assert (
        catchment.catchment_id
        == "C_001"
    )


def test_invalid_curve_number_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "base_curve_number must be between 1 and 100"
        ),
    ):
        CatchmentProfile(
            catchment_id="C_001",
            polygon=make_polygon(),
            base_curve_number=120.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            mean_elevation_m=1200.0,
            provenance=(
                SpatialDataProvenance.DERIVED
            ),
        )


def test_invalid_flow_length_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "flow_length_m must be greater than 0"
        ),
    ):
        CatchmentProfile(
            catchment_id="C_001",
            polygon=make_polygon(),
            base_curve_number=80.0,
            flow_length_m=0.0,
            slope_fraction=0.10,
            mean_elevation_m=1200.0,
            provenance=(
                SpatialDataProvenance.DERIVED
            ),
        )


def test_invalid_slope_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "slope_fraction must be greater than 0"
        ),
    ):
        CatchmentProfile(
            catchment_id="C_001",
            polygon=make_polygon(),
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.0,
            mean_elevation_m=1200.0,
            provenance=(
                SpatialDataProvenance.DERIVED
            ),
        )


def test_drainage_segment_valid():
    drain = DrainageSegment(
        drain_id="D_001",
        geometry=LineString(
            [
                (77.01, 30.01),
                (77.05, 30.05),
            ]
        ),
        catchment_id="C_001",
        provenance=(
            SpatialDataProvenance.VERIFIED
        ),
    )

    assert drain.drain_id == "D_001"
    assert (
        drain.catchment_id
        == "C_001"
    )


def test_road_segment_valid():
    road = RoadSegment(
        road_id="R_001",
        geometry=LineString(
            [
                (77.02, 30.02),
                (77.08, 30.08),
            ]
        ),
        road_name="Test Road",
        catchment_id="C_001",
        provenance=(
            SpatialDataProvenance.VERIFIED
        ),
    )

    assert road.road_id == "R_001"

    assert (
        road.road_name
        == "Test Road"
    )