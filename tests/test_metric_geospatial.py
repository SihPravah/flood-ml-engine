import pytest

from shapely.geometry import LineString

from pravaha_ml.geospatial.metric import (
    line_distance_metres,
    utm_epsg_for_lon_lat,
)


def test_northern_india_uses_utm_zone_43n():
    epsg = utm_epsg_for_lon_lat(
        longitude=77.0,
        latitude=30.0,
    )

    assert epsg == 32643


def test_southern_hemisphere_uses_327_prefix():
    epsg = utm_epsg_for_lon_lat(
        longitude=77.0,
        latitude=-30.0,
    )

    assert epsg == 32743


def test_invalid_longitude_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "longitude must be between -180 and 180"
        ),
    ):
        utm_epsg_for_lon_lat(
            longitude=200.0,
            latitude=30.0,
        )


def test_unsupported_polar_latitude_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "UTM helper supports latitudes"
        ),
    ):
        utm_epsg_for_lon_lat(
            longitude=77.0,
            latitude=88.0,
        )


def test_identical_lines_have_zero_distance():
    line = LineString(
        [
            (77.0000, 30.0000),
            (77.0010, 30.0010),
        ]
    )

    result = line_distance_metres(
        line,
        line,
    )

    assert (
        result.distance_m
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )


def test_separated_lines_have_positive_metric_distance():
    first = LineString(
        [
            (77.0000, 30.0000),
            (77.0010, 30.0000),
        ]
    )

    second = LineString(
        [
            (77.0000, 30.0010),
            (77.0010, 30.0010),
        ]
    )

    result = line_distance_metres(
        first,
        second,
    )

    assert result.distance_m > 50.0
    assert result.distance_m < 150.0

    assert (
        result.projected_crs_epsg
        == 32643
    )