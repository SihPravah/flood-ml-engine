from datetime import datetime, timezone

from pravaha_ml.demo.ids import (
    DEMO_CATCHMENT_ID,
    DEMO_ROAD_DIRECT_ID,
)
from pravaha_ml.geospatial.models import SpatialDataProvenance
from pravaha_ml.geospatial.static_context import (
    METADATA,
    build_chandrabani_predictor_context,
)
from pravaha_ml.inference.predictor import (
    RouteStatus,
    predict_intelligence,
)


def fused_state() -> dict:
    def window(value: float) -> dict:
        return {
            "value_mm": value,
            "status": "DERIVED",
            "coverage_fraction": 1.0,
            "largest_gap_minutes": 5.0,
            "latest_observation_age_minutes": 2.0,
            "observation_count": 6,
            "quality": "GOOD",
        }

    return {
        "catchment_id": DEMO_CATCHMENT_ID,
        "state_time": datetime(2026, 9, 9, 9, 0, tzinfo=timezone.utc).isoformat(),
        "rainfall": {
            "intensity": {
                "value": 48.0,
                "status": "SIMULATED",
                "confidence": 0.90,
                "age_minutes": 2.0,
            },
            "rain_15m": window(12.0),
            "rain_30m": window(24.0),
            "rain_1h": window(48.0),
            "rain_3h": window(92.0),
            "rain_6h": window(118.0),
            "rain_24h": window(145.0),
        },
        "soil": {
            "saturation": 0.82,
            "status": "SIMULATED",
            "confidence": 0.86,
            "age_minutes": 2.0,
        },
        "data_quality": {
            "overall_score": 0.90,
            "missing_sources": [],
            "temporal_freshness": "GOOD",
        },
    }


def test_chandrabani_context_preserves_static_source_statuses():
    context = build_chandrabani_predictor_context()

    assert context.static.terrain_source == METADATA.dem_source
    assert context.static.terrain_source_status == "DERIVED_FROM_REAL_DATA"
    assert context.static.road_source_status == "OPEN_REAL_DATA"
    assert context.static.drain_capacity_status == "ESTIMATED"
    assert context.static.historical_inventory_provenance.value == "MISSING"
    assert context.evacuation.exposure is not None
    assert context.evacuation.exposure.population is None
    assert {
        road.road_id: road.provenance
        for road in context.roads.roads
    }[DEMO_ROAD_DIRECT_ID] == SpatialDataProvenance.OPEN_REAL_DATA


def test_predictor_runs_with_real_chandrabani_static_context():
    result = predict_intelligence(
        fused_state=fused_state(),
        context=build_chandrabani_predictor_context(),
    )

    assert result.hydrology.concentration_time_minutes > 0.0
    assert result.drainage is not None
    assert result.roads
    assert result.route.status == RouteStatus.ROUTE_FOUND
    assert result.city.summary.catchment_count == 1
    assert "drainage_context_missing" not in result.limitations
    assert "road_context_missing" not in result.limitations


def test_no_safe_route_stays_explicit_with_chandrabani_context():
    result = predict_intelligence(
        fused_state=fused_state(),
        context=build_chandrabani_predictor_context(no_safe_route=True),
    )

    assert result.route.status == RouteStatus.NO_SAFE_ROUTE
