from datetime import datetime, timedelta, timezone

import pytest
from shapely.geometry import LineString

from pravaha_ml.anticipation.models import RiskObservation
from pravaha_ml.city.models import CityOperationalStatus
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainRiskLevel,
    DrainStaticProfile,
)
from pravaha_ml.evacuation.clearance_models import (
    EvacuationFlowPath,
)
from pravaha_ml.evacuation.models import (
    ShelterOption,
    ShelterStatus,
)
from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
)
from pravaha_ml.geospatial.models import (
    DrainageSegment,
    RoadSegment,
    SpatialDataProvenance,
)
from pravaha_ml.impact.models import (
    AdministrativeExposure,
    AdministrativeUnitType,
)
from pravaha_ml.inference.confidence import (
    PredictionDisposition,
)
from pravaha_ml.inference.fused_state_adapter import (
    DataStatus,
    adapt_fused_catchment_state,
)
from pravaha_ml.inference.predictor import (
    DrainageContext,
    EvacuationContext,
    PredictorContext,
    RoadContext,
    RoadEnvironmentalContext,
    RouteEdgeSpec,
    RouteStatus,
    StaticCatchmentContext,
    predict_intelligence,
)
from pravaha_ml.landslide.models import (
    LandslideRiskLevel,
)
from pravaha_ml.roads.models import (
    RoadRecommendation,
)


BASE_TIME = datetime(
    2026,
    9,
    9,
    8,
    0,
    tzinfo=timezone.utc,
)


def fused_state(
    *,
    rain_1h: float = 48.0,
    rain_3h: float = 92.0,
    rain_24h: float = 145.0,
    soil: float = 0.82,
    status: str = "SIMULATED",
    quality: str = "GOOD",
    missing_1h: bool = False,
    state_time: datetime = BASE_TIME,
) -> dict:
    def window(value: float, minutes: float) -> dict:
        return {
            "value_mm": value,
            "status": "MISSING" if missing_1h and minutes == 60 else "DERIVED",
            "coverage_fraction": 1.0,
            "largest_gap_minutes": 5.0,
            "latest_observation_age_minutes": 2.0,
            "observation_count": 6,
            "quality": quality if minutes == 60 else "GOOD",
        }

    return {
        "catchment_id": "UK-CHM-DEHRADUN-01",
        "state_time": state_time.isoformat(),
        "rainfall": {
            "intensity": {
                "value": rain_1h,
                "status": status,
                "confidence": 0.92,
                "age_minutes": 2.0,
            },
            "rain_15m": window(rain_1h / 4.0, 15),
            "rain_30m": window(rain_1h / 2.0, 30),
            "rain_1h": window(rain_1h, 60),
            "rain_3h": window(rain_3h, 180),
            "rain_6h": window(rain_3h + 10.0, 360),
            "rain_24h": window(rain_24h, 1440),
        },
        "soil": {
            "saturation": soil,
            "status": status,
            "confidence": 0.88,
            "age_minutes": 3.0,
        },
        "data_quality": {
            "overall_score": 0.90,
            "missing_sources": [],
            "temporal_freshness": quality,
        },
    }


def full_context(
    *,
    blocked_route: bool = False,
    authority_closed: bool = False,
    landslide_prone: bool = True,
) -> PredictorContext:
    drain_profile = DrainStaticProfile(
        drain_id="DRAIN-01",
        width_m=0.28,
        depth_m=0.25,
        slope_fraction=0.015,
        manning_roughness=0.030,
        blockage_fraction=0.35,
        condition=DrainCondition.DEGRADED,
        capacity_provenance=DrainCapacityProvenance.ESTIMATED,
        catchment_id="UK-CHM-DEHRADUN-01",
    )
    drain = DrainageSegment(
        drain_id="DRAIN-01",
        geometry=LineString(
            [
                (78.0300, 30.3200),
                (78.0330, 30.3230),
            ]
        ),
        provenance=SpatialDataProvenance.ESTIMATED,
        catchment_id="UK-CHM-DEHRADUN-01",
    )
    roads = (
        RoadSegment(
            road_id="ROAD-FAST",
            geometry=LineString(
                [
                    (78.0300, 30.3202),
                    (78.0330, 30.3232),
                ]
            ),
            road_name="Fast Road",
            provenance=SpatialDataProvenance.ESTIMATED,
            catchment_id="UK-CHM-DEHRADUN-01",
        ),
        RoadSegment(
            road_id="ROAD-BYPASS",
            geometry=LineString(
                [
                    (78.0290, 30.3190),
                    (78.0340, 30.3245),
                ]
            ),
            road_name="Bypass Road",
            provenance=SpatialDataProvenance.ESTIMATED,
            catchment_id="UK-CHM-DEHRADUN-01",
        ),
    )
    environmental = {
        "ROAD-FAST": RoadEnvironmentalContext(
            catchment_risk_score=0.0,
            terrain_depression_score=0.55,
            stream_proximity_score=0.60,
            historical_waterlogging_score=0.85,
            road_surface_vulnerability_score=0.80,
            data_confidence=0.82,
            authority_closed=authority_closed,
        ),
        "ROAD-BYPASS": RoadEnvironmentalContext(
            catchment_risk_score=0.0,
            terrain_depression_score=0.15,
            stream_proximity_score=0.20,
            historical_waterlogging_score=0.05,
            road_surface_vulnerability_score=0.10,
            data_confidence=0.86,
        ),
    }
    if blocked_route:
        environmental = {
            road_id: RoadEnvironmentalContext(
                catchment_risk_score=item.catchment_risk_score,
                terrain_depression_score=0.95,
                stream_proximity_score=0.95,
                historical_waterlogging_score=0.95,
                road_surface_vulnerability_score=0.95,
                data_confidence=item.data_confidence,
                authority_closed=item.authority_closed,
            )
            for road_id, item in environmental.items()
        }

    return PredictorContext(
        static=StaticCatchmentContext(
            catchment_area_km2=2.4,
            base_curve_number=78.0,
            flow_length_m=1200.0,
            slope_fraction=0.12,
            slope_degrees=34.0 if landslide_prone else 12.0,
            historical_landslide_score=0.75 if landslide_prone else 0.05,
            land_cover_disturbance_score=0.50 if landslide_prone else 0.05,
            static_data_confidence=0.76,
            affected_population=1800,
        ),
        drainage=DrainageContext(
            profiles=(drain_profile,),
            catchment_drain_id="DRAIN-01",
            peaking_factor=1.8,
        ),
        roads=RoadContext(
            roads=roads,
            drains=(drain,),
            environmental_contexts=environmental,
            route_edges=(
                RouteEdgeSpec(
                    "A",
                    "D",
                    "ROAD-FAST",
                    5.0,
                ),
                RouteEdgeSpec(
                    "A",
                    "B",
                    "ROAD-BYPASS",
                    8.0,
                ),
                RouteEdgeSpec(
                    "B",
                    "D",
                    "ROAD-BYPASS",
                    8.0,
                ),
            ),
            start_node="A",
            destination_node="D",
        ),
        evacuation=EvacuationContext(
            exposure=AdministrativeExposure(
                unit_id="WARD-07",
                unit_name="Demo Ward 7",
                unit_type=AdministrativeUnitType.WARD,
                population=1800,
                road_count=2,
                drain_count=1,
                critical_facility_count=1,
                data_confidence=0.80,
            ),
            shelters=(
                ShelterOption(
                    shelter_id="SHELTER-01",
                    shelter_name="School Shelter",
                    total_capacity=2200,
                    current_occupancy=150,
                    status=ShelterStatus.AVAILABLE,
                    data_confidence=0.80,
                ),
            ),
            clearance_paths=(
                EvacuationFlowPath(
                    path_id="PATH-01",
                    route_id="selected_route",
                    shelter_id="SHELTER-01",
                    travel_time_minutes=18.0,
                    route_throughput_people_per_minute=90.0,
                    shelter_intake_people_per_minute=120.0,
                    available_shelter_capacity=2050,
                    data_confidence=0.78,
                ),
            ),
        ),
        risk_history=(
            RiskObservation(
                timestamp=BASE_TIME - timedelta(minutes=45),
                risk_score=0.30,
            ),
            RiskObservation(
                timestamp=BASE_TIME - timedelta(minutes=30),
                risk_score=0.39,
            ),
            RiskObservation(
                timestamp=BASE_TIME - timedelta(minutes=15),
                risk_score=0.49,
            ),
        ),
    )


def test_fused_state_adapter_preserves_v21_shape_and_provenance():
    adapted = adapt_fused_catchment_state(
        fused_state(status="SIMULATED")
    )

    assert adapted.catchment_id == "UK-CHM-DEHRADUN-01"
    assert adapted.rainfall.rain_1h.value_mm == pytest.approx(48.0)
    assert adapted.rainfall.rain_1h.status == DataStatus.DERIVED
    assert adapted.rainfall.intensity.status == DataStatus.SIMULATED
    assert (
        adapted.original_provenance()["rainfall.intensity"]
        == "SIMULATED"
    )


def test_predictor_runs_end_to_end_to_city_intelligence():
    result = predict_intelligence(
        fused_state=fused_state(),
        context=full_context(),
    )

    assert result.hydrology.runoff_mm > 0.0
    assert result.flood_prediction.risk_score > 0.0
    assert result.confidence.overall_confidence > 0.0
    assert result.landslide.susceptibility_score > 0.0
    assert result.drainage is not None
    assert len(result.roads) == 2
    assert result.route.status == RouteStatus.ROUTE_FOUND
    assert result.ward_impact is not None
    assert result.evacuation_readiness is not None
    assert result.city.summary.catchment_count == 1
    assert result.to_map_dto()["catchment"]["fused_state"] == (
        "FusedCatchmentState v2.1"
    )


def test_low_risk_low_confidence_is_not_safe():
    result = predict_intelligence(
        fused_state=fused_state(
            rain_1h=0.0,
            rain_3h=0.0,
            rain_24h=0.0,
            soil=0.05,
            quality="UNUSABLE",
        ),
        context=full_context(landslide_prone=False),
    )

    assert result.flood_prediction.risk_level.value == "LOW"
    assert (
        result.confidence.disposition
        == PredictionDisposition.INSUFFICIENT_DATA
    )
    assert (
        result.confidence.can_treat_low_risk_as_reliable
        is False
    )


def test_simulated_is_not_treated_as_observed():
    result = predict_intelligence(
        fused_state=fused_state(status="SIMULATED"),
        context=full_context(),
    )

    assert (
        result.fused_state.original_provenance()["soil.saturation"]
        == "SIMULATED"
    )
    assert "estimated_inputs_present" in result.confidence.reasons


def test_missing_rainfall_is_not_zero_reliable_data():
    result = predict_intelligence(
        fused_state=fused_state(
            rain_1h=0.0,
            missing_1h=True,
        ),
        context=full_context(),
    )

    assert result.hydrology.rain_1h_mm == pytest.approx(0.0)
    assert result.fused_state.rainfall.rain_1h.status == DataStatus.MISSING
    assert result.confidence.disposition == (
        PredictionDisposition.INSUFFICIENT_DATA
    )
    assert "missing_rainfall_window" in result.limitations


def test_unusable_window_blocks_reliable_prediction():
    result = predict_intelligence(
        fused_state=fused_state(quality="UNUSABLE"),
        context=full_context(),
    )

    assert result.fused_state.temporal_quality_report().level == (
        TemporalQualityLevel.UNUSABLE
    )
    assert result.flood_prediction.model_name == "insufficient_data_gate"
    assert result.confidence.disposition == (
        PredictionDisposition.INSUFFICIENT_DATA
    )


def test_closed_and_avoid_remain_distinct():
    result = predict_intelligence(
        fused_state=fused_state(),
        context=full_context(authority_closed=True),
    )

    recommendations = {
        road.road.road_id: road.assessment.recommendation
        for road in result.roads
    }
    assert recommendations["ROAD-FAST"] == RoadRecommendation.CLOSED
    assert recommendations["ROAD-BYPASS"] != RoadRecommendation.CLOSED


def test_no_safe_route_is_explicit_domain_result():
    result = predict_intelligence(
        fused_state=fused_state(
            rain_1h=78.0,
            rain_3h=140.0,
            rain_24h=220.0,
            soil=0.95,
        ),
        context=full_context(blocked_route=True),
    )

    assert result.route.status == RouteStatus.NO_SAFE_ROUTE
    assert result.route.reason_code == "NO_ROUTABLE_PATH"
    assert result.route.route is None


def test_landslide_can_degrade_road_viability():
    prone = predict_intelligence(
        fused_state=fused_state(
            rain_1h=70.0,
            rain_3h=135.0,
            rain_24h=210.0,
            soil=0.96,
        ),
        context=full_context(landslide_prone=True),
    )
    stable = predict_intelligence(
        fused_state=fused_state(
            rain_1h=70.0,
            rain_3h=135.0,
            rain_24h=210.0,
            soil=0.96,
        ),
        context=full_context(landslide_prone=False),
    )

    assert prone.landslide.risk_level in {
        LandslideRiskLevel.WARNING,
        LandslideRiskLevel.HIGH,
        LandslideRiskLevel.SEVERE,
    }
    assert max(
        road.assessment.risk_score
        for road in prone.roads
    ) >= max(
        road.assessment.risk_score
        for road in stable.roads
    )
    assert any(
        "landslide_susceptibility_degrades_road_viability"
        in road.assessment.reasons
        for road in prone.roads
    )


def test_worsening_scenario_escalates_city_state():
    normal = predict_intelligence(
        fused_state=fused_state(
            rain_1h=2.0,
            rain_3h=4.0,
            rain_24h=8.0,
            soil=0.25,
        ),
        context=full_context(landslide_prone=False),
    )
    severe = predict_intelligence(
        fused_state=fused_state(
            rain_1h=86.0,
            rain_3h=165.0,
            rain_24h=260.0,
            soil=0.97,
        ),
        context=full_context(blocked_route=True),
    )

    assert normal.city.summary.operational_status in {
        CityOperationalStatus.NORMAL,
        CityOperationalStatus.ELEVATED,
    }
    assert severe.city.summary.operational_status in {
        CityOperationalStatus.ELEVATED,
        CityOperationalStatus.EMERGENCY,
    }
    assert severe.flood_prediction.risk_score > (
        normal.flood_prediction.risk_score
    )
    assert severe.city.summary.roads_to_avoid >= (
        normal.city.summary.roads_to_avoid
    )
