from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString

from pravaha_ml.demo.ids import (
    DEMO_CATCHMENT_ID,
    DEMO_DRAIN_ID,
    DEMO_ROAD_BYPASS_ID,
    DEMO_ROAD_CLOSED_ID,
    DEMO_ROAD_DIRECT_ID,
    DEMO_SHELTER_ID,
    DEMO_WARD_ID,
)
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainStaticProfile,
)
from pravaha_ml.evacuation.clearance_models import EvacuationFlowPath
from pravaha_ml.evacuation.models import (
    EvacuationRouteOption,
    EvacuationRouteStatus,
    ShelterOption,
    ShelterStatus,
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
from pravaha_ml.inference.predictor import (
    DrainageContext,
    EvacuationContext,
    PredictorContext,
    RoadContext,
    RoadEnvironmentalContext,
    RouteEdgeSpec,
    StaticCatchmentContext,
)
from pravaha_ml.landslide.models import LandslideDataProvenance


@dataclass(frozen=True)
class ChandrabaniGISMetadata:
    study_area_id: str = "DEHRADUN-CHANDRABANI-PS26192"
    name: str = "Chandrabani focused micro-catchment study area"
    public_crs: str = "EPSG:4326"
    metric_crs: str = "EPSG:32643"
    dem_source: str = "OpenTopoData SRTM 30m elevation API"
    dem_status: str = "OPEN_REAL_DATA"
    road_source: str = "OpenStreetMap"
    road_status: str = "OPEN_REAL_DATA"
    stream_source: str = "OpenStreetMap"
    stream_status: str = "OPEN_REAL_DATA"
    municipal_drain_capacity_status: str = "ESTIMATED"
    shelter_designation_status: str = "DEMO"
    landslide_inventory_status: str = "NOT_AVAILABLE"


METADATA = ChandrabaniGISMetadata()

ROAD_DIRECT_COORDS = (
    (77.9921844, 30.2815189),
    (77.9912487, 30.2822485),
    (77.9919438, 30.2838838),
    (77.9926018, 30.2854228),
    (77.9936385, 30.2876034),
    (77.9948480, 30.2898648),
    (77.9974537, 30.2940510),
)
ROAD_BYPASS_COORDS = (
    (77.9889999, 30.2752512),
    (77.9894658, 30.2750446),
    (77.9902560, 30.2745811),
    (77.9918576, 30.2736417),
    (77.9945152, 30.2720267),
    (77.9965798, 30.2700555),
)
ROAD_CLOSED_COORDS = (
    (77.9766333, 30.2810359),
    (77.9767205, 30.2809907),
)
STREAM_COORDS = (
    (77.9883529, 30.2996786),
    (77.9864617, 30.2978559),
    (77.9831561, 30.2950073),
    (77.9815021, 30.2941741),
    (77.9807278, 30.2910940),
    (77.9787451, 30.2892119),
    (77.9768518, 30.2877356),
)


def build_chandrabani_predictor_context(
    *,
    authority_closure_active: bool = False,
    no_safe_route: bool = False,
) -> PredictorContext:
    drain_profile = DrainStaticProfile(
        drain_id=DEMO_DRAIN_ID,
        width_m=0.28,
        depth_m=0.25,
        slope_fraction=0.0079,
        manning_roughness=0.030,
        blockage_fraction=0.35,
        condition=DrainCondition.UNKNOWN,
        capacity_provenance=DrainCapacityProvenance.ESTIMATED,
        catchment_id=DEMO_CATCHMENT_ID,
    )
    drain = DrainageSegment(
        drain_id=DEMO_DRAIN_ID,
        geometry=LineString(STREAM_COORDS),
        provenance=SpatialDataProvenance.ESTIMATED,
        catchment_id=DEMO_CATCHMENT_ID,
    )
    roads = (
        RoadSegment(
            road_id=DEMO_ROAD_DIRECT_ID,
            geometry=LineString(ROAD_DIRECT_COORDS),
            road_name="Transport Nagar Road",
            provenance=SpatialDataProvenance.OPEN_REAL_DATA,
            catchment_id=DEMO_CATCHMENT_ID,
        ),
        RoadSegment(
            road_id=DEMO_ROAD_BYPASS_ID,
            geometry=LineString(ROAD_BYPASS_COORDS),
            road_name="Post Office Road",
            provenance=SpatialDataProvenance.OPEN_REAL_DATA,
            catchment_id=DEMO_CATCHMENT_ID,
        ),
        RoadSegment(
            road_id=DEMO_ROAD_CLOSED_ID,
            geometry=LineString(ROAD_CLOSED_COORDS),
            road_name="Unnamed Road",
            provenance=SpatialDataProvenance.OPEN_REAL_DATA,
            catchment_id=DEMO_CATCHMENT_ID,
        ),
    )
    environmental = {
        DEMO_ROAD_DIRECT_ID: RoadEnvironmentalContext(
            catchment_risk_score=0.0,
            terrain_depression_score=0.42,
            stream_proximity_score=0.66,
            historical_waterlogging_score=0.35,
            road_surface_vulnerability_score=0.28,
            data_confidence=0.76,
            authority_closed=False,
        ),
        DEMO_ROAD_BYPASS_ID: RoadEnvironmentalContext(
            catchment_risk_score=0.0,
            terrain_depression_score=0.18,
            stream_proximity_score=0.30,
            historical_waterlogging_score=0.12,
            road_surface_vulnerability_score=0.18,
            data_confidence=0.78,
        ),
        DEMO_ROAD_CLOSED_ID: RoadEnvironmentalContext(
            catchment_risk_score=0.0,
            terrain_depression_score=0.50,
            stream_proximity_score=0.55,
            historical_waterlogging_score=0.20,
            road_surface_vulnerability_score=0.20,
            data_confidence=0.70,
            authority_closed=authority_closure_active,
        ),
    }
    if no_safe_route:
        environmental = {
            road_id: RoadEnvironmentalContext(
                catchment_risk_score=item.catchment_risk_score,
                terrain_depression_score=0.95,
                stream_proximity_score=0.95,
                historical_waterlogging_score=0.95,
                road_surface_vulnerability_score=0.95,
                data_confidence=item.data_confidence,
                authority_closed=(
                    True if road_id == DEMO_ROAD_CLOSED_ID else item.authority_closed
                ),
            )
            for road_id, item in environmental.items()
        }

    return PredictorContext(
        static=StaticCatchmentContext(
            catchment_area_km2=10.25,
            base_curve_number=79.0,
            flow_length_m=2500.0,
            slope_fraction=0.0079,
            slope_degrees=0.46,
            mean_elevation_m=605.1,
            min_elevation_m=594.0,
            max_elevation_m=646.0,
            terrain_source=METADATA.dem_source,
            terrain_source_status="DERIVED_FROM_REAL_DATA",
            road_source_status=METADATA.road_status,
            stream_source_status=METADATA.stream_status,
            catchment_geometry_status="ESTIMATED",
            admin_source_status=METADATA.road_status,
            shelter_source_status=METADATA.shelter_designation_status,
            drain_capacity_status=METADATA.municipal_drain_capacity_status,
            historical_landslide_score=0.0,
            land_cover_disturbance_score=0.10,
            static_data_confidence=0.78,
            historical_inventory_provenance=LandslideDataProvenance.MISSING,
            affected_population=None,
        ),
        drainage=DrainageContext(
            profiles=(drain_profile,),
            catchment_drain_id=DEMO_DRAIN_ID,
            peaking_factor=1.8,
        ),
        roads=RoadContext(
            roads=roads,
            drains=(drain,),
            environmental_contexts=environmental,
            route_edges=(
                RouteEdgeSpec("ORIGIN", "SHELTER", DEMO_ROAD_DIRECT_ID, 11.0),
                RouteEdgeSpec("ORIGIN", "BYPASS", DEMO_ROAD_BYPASS_ID, 8.0),
                RouteEdgeSpec("BYPASS", "SHELTER", DEMO_ROAD_BYPASS_ID, 10.0),
                RouteEdgeSpec("ORIGIN", "BRIDGE", DEMO_ROAD_CLOSED_ID, 6.0),
                RouteEdgeSpec("BRIDGE", "SHELTER", DEMO_ROAD_DIRECT_ID, 7.0),
            ),
            start_node="ORIGIN",
            destination_node="ISOLATED" if no_safe_route else "SHELTER",
            maximum_drain_distance_m=2500.0,
        ),
        evacuation=EvacuationContext(
            exposure=AdministrativeExposure(
                unit_id=DEMO_WARD_ID,
                unit_name="Chandrabani settlement point",
                unit_type=AdministrativeUnitType.VILLAGE,
                population=None,
                road_count=len(roads),
                drain_count=1,
                critical_facility_count=3,
                data_confidence=0.68,
            ),
            shelters=(
                ShelterOption(
                    shelter_id=DEMO_SHELTER_ID,
                    shelter_name="Demo shelter at Rajaram Mohan Roy Academy POI",
                    total_capacity=None,
                    current_occupancy=None,
                    status=ShelterStatus.UNKNOWN,
                    data_confidence=0.40,
                ),
            ),
            routes=(
                EvacuationRouteOption(
                    route_id="ROUTE-DEMO-001-BYPASS",
                    shelter_id=DEMO_SHELTER_ID,
                    route_status=EvacuationRouteStatus.CAUTION,
                    travel_time_minutes=18.0,
                    maximum_risk_score=0.38,
                    minimum_confidence=0.70,
                ),
            ),
            clearance_paths=(
                EvacuationFlowPath(
                    path_id="PATH-CHANDRABANI-01",
                    route_id="ROUTE-DEMO-001-BYPASS",
                    shelter_id=DEMO_SHELTER_ID,
                    travel_time_minutes=18.0,
                    route_throughput_people_per_minute=90.0,
                    shelter_intake_people_per_minute=120.0,
                    available_shelter_capacity=0,
                    data_confidence=0.40,
                ),
            ),
        ),
    )
