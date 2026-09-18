from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from pravaha_ml.anticipation.models import (
    RiskObservation,
    RiskTrajectoryAssessment,
    ThresholdStatus,
)
from pravaha_ml.anticipation.trajectory import (
    assess_risk_trajectory,
)
from pravaha_ml.cascade.engine import (
    assess_hazard_cascade,
)
from pravaha_ml.cascade.models import (
    HazardCascadeAssessment,
    HazardEvidenceStatus,
    HazardState,
    HazardType,
)
from pravaha_ml.city.models import (
    CatchmentIntelligence,
    CityIntelligenceState,
)
from pravaha_ml.city.orchestrator import (
    build_city_intelligence_state,
)
from pravaha_ml.drainage.flow import (
    CatchmentRunoffInput,
)
from pravaha_ml.drainage.models import (
    DrainStaticProfile,
)
from pravaha_ml.drainage.network import (
    DrainConnection,
    DrainNetworkError,
    DrainNetworkResult,
    route_drainage_network,
)
from pravaha_ml.evacuation.clearance import (
    assess_evacuation_clearance,
)
from pravaha_ml.evacuation.clearance_models import (
    EvacuationClearanceAssessment,
    EvacuationClearanceInputs,
    EvacuationFlowPath,
)
from pravaha_ml.evacuation.models import (
    EvacuationReadinessAssessment,
    EvacuationReadinessInputs,
    EvacuationRouteOption,
    EvacuationRouteStatus,
    ShelterOption,
)
from pravaha_ml.evacuation.readiness import (
    assess_evacuation_readiness,
)
from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.impact.models import (
    AdministrativeExposure,
    RouteReadiness,
    WardHazardInputs,
    WardImpactAssessment,
)
from pravaha_ml.impact.ward import (
    assess_ward_impact,
)
from pravaha_ml.inference.confidence import (
    ConfidenceAssessment,
    PredictionDisposition,
    assess_prediction_confidence,
)
from pravaha_ml.inference.fused_state_adapter import (
    DataStatus,
    FusedCatchmentStateV21,
    FusedRainWindow,
    adapt_fused_catchment_state,
)
from pravaha_ml.landslide.models import (
    LandslideAssessment,
    LandslideDataProvenance,
    LandslideInputs,
)
from pravaha_ml.landslide.risk import (
    assess_landslide_susceptibility,
)
from pravaha_ml.models.baseline import (
    RiskPrediction,
)
from pravaha_ml.models.risk import (
    classify_risk,
)
from pravaha_ml.roads.models import (
    RoadFloodAssessment,
    RoadFloodRiskLevel,
    RoadRecommendation,
)
from pravaha_ml.roads.routing import (
    RoadGraphEdge,
    RouteNotFoundError,
    RouteSafetyLevel,
    SafeRouteResult,
    find_safest_route,
)
from pravaha_ml.roads.spatial import (
    DrainAssociationError,
    RoadEnvironmentalContext,
    SpatialRoadRiskResult,
    assess_road_from_spatial_context,
)
from pravaha_ml.training.serialization import (
    load_model,
)


class RouteStatus(str, Enum):
    ROUTE_FOUND = "ROUTE_FOUND"
    NO_SAFE_ROUTE = "NO_SAFE_ROUTE"
    NOT_EVALUATED = "NOT_EVALUATED"


class ModelRuntimeStatus(str, Enum):
    REAL_ARTIFACT = "REAL_ARTIFACT"
    DEVELOPMENT_FALLBACK = "DEVELOPMENT_FALLBACK"


@dataclass(frozen=True)
class StaticCatchmentContext:
    catchment_area_km2: float
    base_curve_number: float
    flow_length_m: float
    slope_fraction: float
    slope_degrees: float

    mean_elevation_m: float | None = None
    min_elevation_m: float | None = None
    max_elevation_m: float | None = None
    terrain_source: str | None = None
    terrain_source_status: str = "ESTIMATED"
    road_source_status: str = "ESTIMATED"
    stream_source_status: str = "ESTIMATED"
    catchment_geometry_status: str = "ESTIMATED"
    admin_source_status: str = "ESTIMATED"
    shelter_source_status: str = "ESTIMATED"
    drain_capacity_status: str = "ESTIMATED"
    hand_status: str = "NOT_AVAILABLE"
    twi_status: str = "NOT_AVAILABLE"

    dry_threshold_percentage: float = 35.0
    wet_threshold_percentage: float = 70.0
    historical_landslide_score: float = 0.0
    land_cover_disturbance_score: float = 0.0
    static_data_confidence: float = 0.70
    historical_inventory_provenance: (
        LandslideDataProvenance
    ) = LandslideDataProvenance.ESTIMATED
    affected_population: int | None = None


@dataclass(frozen=True)
class DrainageContext:
    profiles: tuple[DrainStaticProfile, ...] = ()
    connections: tuple[DrainConnection, ...] = ()
    catchment_drain_id: str | None = None
    peaking_factor: float = 1.0


@dataclass(frozen=True)
class RouteEdgeSpec:
    start_node: str
    end_node: str
    road_id: str
    travel_time_minutes: float


@dataclass(frozen=True)
class RoadContext:
    roads: tuple[Any, ...] = ()
    drains: tuple[Any, ...] = ()
    environmental_contexts: Mapping[
        str,
        RoadEnvironmentalContext,
    ] | None = None
    route_edges: tuple[RouteEdgeSpec, ...] = ()
    start_node: str | None = None
    destination_node: str | None = None
    maximum_drain_distance_m: float | None = None
    apply_landslide_to_roads: bool = True


@dataclass(frozen=True)
class EvacuationContext:
    exposure: AdministrativeExposure | None = None
    shelters: tuple[ShelterOption, ...] = ()
    routes: tuple[EvacuationRouteOption, ...] = ()
    clearance_paths: tuple[EvacuationFlowPath, ...] = ()
    mobilization_delay_minutes: float = 10.0


@dataclass(frozen=True)
class PredictorContext:
    static: StaticCatchmentContext
    drainage: DrainageContext = DrainageContext()
    roads: RoadContext = RoadContext()
    evacuation: EvacuationContext = EvacuationContext()
    risk_history: tuple[RiskObservation, ...] = ()


@dataclass(frozen=True)
class RouteDomainResult:
    status: RouteStatus
    route: SafeRouteResult | None
    reason_code: str | None
    message: str
    avoided_closed_segments: int = 0
    avoided_predicted_unsafe_segments: int = 0


@dataclass(frozen=True)
class IntegratedIntelligence:
    fused_state: FusedCatchmentStateV21
    hydrology: HydrologyFeatures
    flood_prediction: RiskPrediction
    model_runtime_status: ModelRuntimeStatus
    model_operationally_validated: bool
    confidence: ConfidenceAssessment
    landslide: LandslideAssessment
    trajectory: RiskTrajectoryAssessment
    cascade: HazardCascadeAssessment
    drainage: DrainNetworkResult | None
    roads: tuple[SpatialRoadRiskResult, ...]
    route: RouteDomainResult
    ward_impact: WardImpactAssessment | None
    evacuation_readiness: EvacuationReadinessAssessment | None
    evacuation_clearance: EvacuationClearanceAssessment | None
    city: CityIntelligenceState
    limitations: tuple[str, ...]

    def to_map_dto(self) -> dict[str, Any]:
        generated_at = self.fused_state.state_time
        return {
            "snapshot_id": _snapshot_id(generated_at),
            "generated_at": generated_at.isoformat(),
            "catchment": {
                "catchment_id": self.fused_state.catchment_id,
                "risk_score": self.flood_prediction.risk_score,
                "risk_level": self.flood_prediction.risk_level.value,
                "confidence": self.confidence.overall_confidence,
                "confidence_level": self.confidence.confidence_level.value,
                "disposition": self.confidence.disposition.value,
                "reasons": list(self.confidence.reasons),
                "provenance": (
                    self.fused_state.original_provenance()
                ),
                "last_updated": generated_at.isoformat(),
                "fused_state": "FusedCatchmentState v2.1",
                "hydrology": _plain(self.hydrology),
                "landslide": _plain(self.landslide),
                "anticipation": _plain(self.trajectory),
            },
            "drains": [
                _plain(drain)
                for drain in (
                    self.drainage.drain_states
                    if self.drainage is not None
                    else ()
                )
            ],
            "roads": [
                _plain(road)
                for road in self.roads
            ],
            "route": _plain(self.route),
            "city": _plain(self.city.summary),
            "limitations": list(self.limitations),
            "model_metadata": {
                "prediction_id": _prediction_id(
                    self.fused_state.catchment_id,
                    generated_at,
                ),
                "model_version": self.flood_prediction.model_version,
                "generated_at": generated_at.isoformat(),
                "input_state_time": self.fused_state.state_time.isoformat(),
                "risk_score": self.flood_prediction.risk_score,
                "risk_level": self.flood_prediction.risk_level.value,
                "confidence": self.confidence.overall_confidence,
                "data_quality_score": self.fused_state.data_quality.overall_score,
                "runtime_status": self.model_runtime_status.value,
                "operationally_validated": self.model_operationally_validated,
                "top_factors": list(self.confidence.reasons[:5]),
            },
            "model": {
                "name": self.flood_prediction.model_name,
                "version": self.flood_prediction.model_version,
                "runtime_status": self.model_runtime_status.value,
                "operationally_validated": (
                    self.model_operationally_validated
                ),
            },
        }


class FloodRiskModel(Protocol):
    def predict(
        self,
        features: HydrologyFeatures,
    ) -> RiskPrediction:
        ...


@dataclass(frozen=True)
class ModelBundle:
    model: FloodRiskModel
    runtime_status: ModelRuntimeStatus
    operationally_validated: bool


class DevelopmentRiskModel:
    model_name = "development_heuristic"
    model_version = "synthetic-development-v1"

    def predict(
        self,
        features: HydrologyFeatures,
    ) -> RiskPrediction:
        score = (
            0.05
            + min(features.rain_1h_mm / 65.0, 1.0) * 0.32
            + min(features.rain_3h_mm / 140.0, 1.0) * 0.12
            + min(features.rain_24h_mm / 220.0, 1.0) * 0.08
            + features.soil_saturation * 0.18
            + features.runoff_ratio * 0.20
            + min(features.api_mm / 220.0, 1.0) * 0.05
        )
        score = max(0.0, min(float(score), 1.0))
        return RiskPrediction(
            risk_score=score,
            risk_level=classify_risk(score),
            model_name=self.model_name,
            model_version=self.model_version,
        )


class ModelProvider:
    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        allow_development_fallback: bool = True,
    ) -> None:
        self.model_path = model_path
        self.allow_development_fallback = allow_development_fallback

    def load(self) -> ModelBundle:
        if self.model_path is not None:
            return ModelBundle(
                model=load_model(self.model_path),
                runtime_status=ModelRuntimeStatus.REAL_ARTIFACT,
                operationally_validated=True,
            )

        if not self.allow_development_fallback:
            raise FileNotFoundError(
                "No trained model artifact configured."
            )

        return ModelBundle(
            model=DevelopmentRiskModel(),
            runtime_status=(
                ModelRuntimeStatus.DEVELOPMENT_FALLBACK
            ),
            operationally_validated=False,
        )


def predict_intelligence(
    *,
    fused_state: Mapping[str, Any] | Any,
    context: PredictorContext,
    model_provider: ModelProvider | None = None,
) -> IntegratedIntelligence:
    fused = adapt_fused_catchment_state(fused_state)
    limitations: list[str] = []

    hydrology = _build_live_hydrology_features(
        fused=fused,
        context=context.static,
        limitations=limitations,
    )

    temporal_quality = fused.temporal_quality_report()
    model_bundle = (
        model_provider or ModelProvider()
    ).load()

    if temporal_quality.can_predict:
        prediction = model_bundle.model.predict(
            hydrology
        )
        model_scores = [prediction.risk_score]
    else:
        prediction = RiskPrediction(
            risk_score=0.0,
            risk_level=classify_risk(0.0),
            model_name="insufficient_data_gate",
            model_version="no-reliable-live-inference",
        )
        model_scores = []
        limitations.append(
            "temporal_or_source_quality_blocks_reliable_prediction"
        )

    confidence = assess_prediction_confidence(
        temporal_quality=temporal_quality,
        source_availability_fraction=(
            fused.source_availability_fraction()
        ),
        input_provenances=fused.confidence_provenances(),
        model_scores=model_scores,
    )

    landslide = _assess_landslide(
        fused=fused,
        hydrology=hydrology,
        confidence=confidence,
        context=context.static,
    )

    trajectory = _assess_trajectory(
        current_time=fused.state_time,
        risk_score=prediction.risk_score,
        risk_history=context.risk_history,
    )

    drainage = _assess_drainage(
        fused=fused,
        hydrology=hydrology,
        confidence=confidence,
        context=context,
        limitations=limitations,
    )

    roads = _assess_roads(
        prediction=prediction,
        confidence=confidence,
        landslide=landslide,
        drainage=drainage,
        context=context,
        limitations=limitations,
    )

    cascade = _assess_cascade(
        fused=fused,
        prediction=prediction,
        confidence=confidence,
        landslide=landslide,
        drainage=drainage,
        roads=roads,
    )

    route = _assess_route(
        roads=roads,
        context=context.roads,
        limitations=limitations,
    )

    ward_impact = _assess_impact(
        prediction=prediction,
        confidence=confidence,
        landslide=landslide,
        cascade=cascade,
        roads=roads,
        drainage=drainage,
        route=route,
        context=context,
    )

    evacuation_readiness = _assess_evacuation_readiness(
        ward_impact=ward_impact,
        trajectory=trajectory,
        route=route,
        context=context.evacuation,
    )

    evacuation_clearance = _assess_evacuation_clearance(
        ward_impact=ward_impact,
        trajectory=trajectory,
        context=context.evacuation,
    )

    catchment = CatchmentIntelligence(
        catchment_id=fused.catchment_id,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level.value,
        confidence=confidence,
        hydrology=hydrology,
        affected_population=(
            context.static.affected_population
        ),
    )

    city = build_city_intelligence_state(
        generated_at=fused.state_time,
        catchments=(catchment,),
        drains=(
            drainage.drain_states
            if drainage is not None
            else ()
        ),
        roads=roads,
    )

    return IntegratedIntelligence(
        fused_state=fused,
        hydrology=hydrology,
        flood_prediction=prediction,
        model_runtime_status=model_bundle.runtime_status,
        model_operationally_validated=(
            model_bundle.operationally_validated
        ),
        confidence=confidence,
        landslide=landslide,
        trajectory=trajectory,
        cascade=cascade,
        drainage=drainage,
        roads=roads,
        route=route,
        ward_impact=ward_impact,
        evacuation_readiness=evacuation_readiness,
        evacuation_clearance=evacuation_clearance,
        city=city,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def _build_live_hydrology_features(
    *,
    fused: FusedCatchmentStateV21,
    context: StaticCatchmentContext,
    limitations: list[str],
) -> HydrologyFeatures:
    rain_15m = _window_value(fused.rainfall.rain_15m, limitations)
    rain_30m = _window_value(fused.rainfall.rain_30m, limitations)
    rain_1h = _window_value(fused.rainfall.rain_1h, limitations)
    rain_3h = _window_value(fused.rainfall.rain_3h, limitations)
    rain_6h = _window_value(fused.rainfall.rain_6h, limitations)
    rain_24h = _window_value(fused.rainfall.rain_24h, limitations)

    soil_saturation = fused.soil.saturation
    if (
        soil_saturation is None
        or fused.soil.status == DataStatus.MISSING
    ):
        limitations.append("missing_soil_saturation")
        soil_saturation = 0.0

    soil_moisture_percentage = soil_saturation * 100.0
    from pravaha_ml.hydrology.concentration_time import (
        calculate_concentration_time,
    )
    from pravaha_ml.hydrology.soil_state import (
        calculate_soil_adjusted_runoff,
    )

    runoff = calculate_soil_adjusted_runoff(
        rainfall_mm=rain_1h,
        base_curve_number=context.base_curve_number,
        soil_moisture_percentage=soil_moisture_percentage,
        dry_threshold_percentage=(
            context.dry_threshold_percentage
        ),
        wet_threshold_percentage=(
            context.wet_threshold_percentage
        ),
    )
    concentration = calculate_concentration_time(
        flow_length_m=context.flow_length_m,
        slope_fraction=context.slope_fraction,
    )
    return HydrologyFeatures(
        rain_15m_mm=rain_15m,
        rain_30m_mm=rain_30m,
        rain_1h_mm=rain_1h,
        rain_3h_mm=rain_3h,
        rain_6h_mm=rain_6h,
        rain_24h_mm=rain_24h,
        api_mm=rain_24h,
        soil_moisture_percentage=(
            soil_moisture_percentage
        ),
        soil_saturation=soil_saturation,
        moisture_condition=(
            runoff.moisture_condition.value
        ),
        base_curve_number=context.base_curve_number,
        effective_curve_number=(
            runoff.effective_curve_number
        ),
        runoff_mm=runoff.runoff.runoff_mm,
        runoff_ratio=runoff.runoff.runoff_ratio,
        flow_length_m=context.flow_length_m,
        slope_fraction=context.slope_fraction,
        concentration_time_minutes=(
            concentration.concentration_time_minutes
        ),
    )


def _window_value(
    window: FusedRainWindow,
    limitations: list[str],
) -> float:
    if window.status == DataStatus.MISSING:
        limitations.append("missing_rainfall_window")
        return 0.0
    if window.value_mm is None:
        limitations.append("null_rainfall_window_value")
        return 0.0
    return window.value_mm


def _assess_landslide(
    *,
    fused: FusedCatchmentStateV21,
    hydrology: HydrologyFeatures,
    confidence: ConfidenceAssessment,
    context: StaticCatchmentContext,
) -> LandslideAssessment:
    return assess_landslide_susceptibility(
        LandslideInputs(
            zone_id=f"{fused.catchment_id}:slope",
            slope_degrees=context.slope_degrees,
            soil_saturation=hydrology.soil_saturation,
            rain_1h_mm=hydrology.rain_1h_mm,
            rain_24h_mm=hydrology.rain_24h_mm,
            api_mm=hydrology.api_mm,
            historical_landslide_score=(
                context.historical_landslide_score
            ),
            land_cover_disturbance_score=(
                context.land_cover_disturbance_score
            ),
            data_confidence=min(
                confidence.overall_confidence,
                context.static_data_confidence,
            ),
            historical_inventory_provenance=(
                context.historical_inventory_provenance
            ),
        )
    )


def _assess_trajectory(
    *,
    current_time: datetime,
    risk_score: float,
    risk_history: Iterable[RiskObservation],
) -> RiskTrajectoryAssessment:
    observations = tuple(risk_history) + (
        RiskObservation(
            timestamp=current_time,
            risk_score=risk_score,
        ),
    )
    return assess_risk_trajectory(
        observations=observations,
        current_time=current_time,
    )


def _assess_drainage(
    *,
    fused: FusedCatchmentStateV21,
    hydrology: HydrologyFeatures,
    confidence: ConfidenceAssessment,
    context: PredictorContext,
    limitations: list[str],
) -> DrainNetworkResult | None:
    drainage_context = context.drainage
    if (
        not drainage_context.profiles
        or drainage_context.catchment_drain_id is None
    ):
        limitations.append("drainage_context_missing")
        return None

    runoff = CatchmentRunoffInput(
        catchment_id=fused.catchment_id,
        drain_id=drainage_context.catchment_drain_id,
        catchment_area_km2=(
            context.static.catchment_area_km2
        ),
        runoff_mm=hydrology.runoff_mm,
        response_time_minutes=(
            hydrology.concentration_time_minutes
        ),
        data_confidence=confidence.overall_confidence,
        peaking_factor=drainage_context.peaking_factor,
    )

    try:
        return route_drainage_network(
            profiles=drainage_context.profiles,
            connections=drainage_context.connections,
            runoff_inputs=(runoff,),
        )
    except DrainNetworkError as exc:
        limitations.append(f"drainage_network_unavailable:{exc}")
        return None


def _assess_roads(
    *,
    prediction: RiskPrediction,
    confidence: ConfidenceAssessment,
    landslide: LandslideAssessment,
    drainage: DrainNetworkResult | None,
    context: PredictorContext,
    limitations: list[str],
) -> tuple[SpatialRoadRiskResult, ...]:
    road_context = context.roads
    if (
        drainage is None
        or not road_context.roads
        or not road_context.drains
        or not road_context.environmental_contexts
    ):
        limitations.append("road_context_missing")
        return ()

    results: list[SpatialRoadRiskResult] = []
    for road in road_context.roads:
        environmental = road_context.environmental_contexts.get(
            road.road_id
        )
        if environmental is None:
            limitations.append(
                f"road_environment_missing:{road.road_id}"
            )
            continue

        enriched = _apply_landslide_to_road_context(
            environmental,
            prediction.risk_score,
            confidence,
            landslide,
            road_context.apply_landslide_to_roads,
        )

        try:
            result = assess_road_from_spatial_context(
                road=road,
                drains=road_context.drains,
                drain_flow_states=drainage.drain_states,
                context=enriched,
                maximum_drain_distance_m=(
                    road_context.maximum_drain_distance_m
                ),
            )
            results.append(
                _tag_landslide_road_reason(result, landslide)
            )
        except DrainAssociationError as exc:
            limitations.append(
                f"road_drain_association_unavailable:{road.road_id}:{exc}"
            )

    return tuple(results)


def _apply_landslide_to_road_context(
    environmental: RoadEnvironmentalContext,
    flood_risk_score: float,
    confidence: ConfidenceAssessment,
    landslide: LandslideAssessment,
    enabled: bool,
) -> RoadEnvironmentalContext:
    vulnerability = environmental.road_surface_vulnerability_score
    terrain = environmental.terrain_depression_score

    if enabled:
        vulnerability = max(
            vulnerability,
            landslide.susceptibility_score,
        )
        terrain = max(
            terrain,
            landslide.slope_component,
        )

    return replace(
        environmental,
        catchment_risk_score=flood_risk_score,
        road_surface_vulnerability_score=vulnerability,
        terrain_depression_score=terrain,
        data_confidence=min(
            environmental.data_confidence,
            confidence.overall_confidence,
            landslide.confidence,
        ),
    )


def _tag_landslide_road_reason(
    result: SpatialRoadRiskResult,
    landslide: LandslideAssessment,
) -> SpatialRoadRiskResult:
    if landslide.susceptibility_score < 0.60:
        return result

    assessment = replace(
        result.assessment,
        reasons=tuple(
            dict.fromkeys(
                result.assessment.reasons
                + (
                    "landslide_susceptibility_degrades_road_viability",
                )
            )
        ),
    )
    return replace(
        result,
        assessment=assessment,
    )


def _assess_route(
    *,
    roads: tuple[SpatialRoadRiskResult, ...],
    context: RoadContext,
    limitations: list[str],
) -> RouteDomainResult:
    if (
        not context.route_edges
        or context.start_node is None
        or context.destination_node is None
    ):
        limitations.append("routing_context_missing")
        return RouteDomainResult(
            status=RouteStatus.NOT_EVALUATED,
            route=None,
            reason_code="ROUTING_CONTEXT_MISSING",
            message="Routing context was not supplied.",
        )

    road_map = {
        result.road.road_id: result.assessment
        for result in roads
    }

    edges: list[RoadGraphEdge] = []
    for edge in context.route_edges:
        assessment = road_map.get(edge.road_id)
        if assessment is None:
            limitations.append(
                f"route_edge_assessment_missing:{edge.road_id}"
            )
            continue
        edges.append(
            RoadGraphEdge(
                start_node=edge.start_node,
                end_node=edge.end_node,
                road_id=edge.road_id,
                travel_time_minutes=edge.travel_time_minutes,
                assessment=assessment,
            )
        )

    if not edges:
        return RouteDomainResult(
            status=RouteStatus.NO_SAFE_ROUTE,
            route=None,
            reason_code="NO_ROUTABLE_EDGE_ASSESSMENTS",
            message=(
                "No route edges had corresponding road-risk "
                "assessments."
            ),
        )

    try:
        route = find_safest_route(
            start_node=context.start_node,
            destination_node=context.destination_node,
            edges=edges,
        )
        return RouteDomainResult(
            status=RouteStatus.ROUTE_FOUND,
            route=route,
            reason_code=None,
            message="A risk-aware route was found.",
            avoided_closed_segments=(
                route.avoided_closed_segments
            ),
            avoided_predicted_unsafe_segments=(
                route.avoided_predicted_unsafe_segments
            ),
        )
    except RouteNotFoundError as exc:
        closed = sum(
            edge.assessment.recommendation
            == RoadRecommendation.CLOSED
            for edge in edges
        )
        avoid = sum(
            edge.assessment.recommendation
            == RoadRecommendation.AVOID
            for edge in edges
        )
        return RouteDomainResult(
            status=RouteStatus.NO_SAFE_ROUTE,
            route=None,
            reason_code="NO_ROUTABLE_PATH",
            message=str(exc),
            avoided_closed_segments=closed,
            avoided_predicted_unsafe_segments=avoid,
        )


def _assess_cascade(
    *,
    fused: FusedCatchmentStateV21,
    prediction: RiskPrediction,
    confidence: ConfidenceAssessment,
    landslide: LandslideAssessment,
    drainage: DrainNetworkResult | None,
    roads: tuple[SpatialRoadRiskResult, ...],
) -> HazardCascadeAssessment:
    hazards: list[HazardState] = [
        HazardState(
            hazard_type=HazardType.EXTREME_RAINFALL,
            intensity_score=min(
                fused.rainfall.rain_1h.value_mm or 0.0,
                75.0,
            )
            / 75.0,
            confidence=confidence.overall_confidence,
            evidence_status=_measurement_evidence_status(
                fused.rainfall.rain_1h.status
            ),
            entity_id=fused.catchment_id,
            reasons=("rain_1h_from_fused_state",),
        ),
        HazardState(
            hazard_type=HazardType.SOIL_SATURATION,
            intensity_score=fused.soil.saturation or 0.0,
            confidence=confidence.overall_confidence,
            evidence_status=_measurement_evidence_status(
                fused.soil.status
            ),
            entity_id=fused.catchment_id,
            reasons=("soil_saturation_from_fused_state",),
        ),
        HazardState(
            hazard_type=HazardType.FLASH_FLOOD,
            intensity_score=prediction.risk_score,
            confidence=confidence.overall_confidence,
            evidence_status=HazardEvidenceStatus.PREDICTED,
            entity_id=fused.catchment_id,
            reasons=("flash_flood_model_prediction",),
        ),
        HazardState(
            hazard_type=HazardType.LANDSLIDE,
            intensity_score=landslide.susceptibility_score,
            confidence=landslide.confidence,
            evidence_status=HazardEvidenceStatus.PREDICTED,
            entity_id=landslide.zone_id,
            reasons=landslide.reasons,
        ),
    ]

    if drainage is not None:
        for drain in drainage.drain_states:
            hazards.append(
                HazardState(
                    hazard_type=HazardType.DRAIN_OVERLOAD,
                    intensity_score=min(
                        drain.capacity_utilization,
                        1.0,
                    ),
                    confidence=drain.data_confidence,
                    evidence_status=(
                        HazardEvidenceStatus.PREDICTED
                    ),
                    entity_id=drain.drain_id,
                    reasons=drain.risk.reasons,
                )
            )

    for road in roads:
        hazards.append(
            HazardState(
                hazard_type=HazardType.ROAD_FLOODING,
                intensity_score=road.assessment.risk_score,
                confidence=road.assessment.data_confidence,
                evidence_status=(
                    HazardEvidenceStatus.PREDICTED
                    if not road.assessment.authoritative_closure
                    else HazardEvidenceStatus.AUTHORITY_CONFIRMED
                ),
                entity_id=road.road.road_id,
                reasons=road.assessment.reasons,
            )
        )

    return assess_hazard_cascade(
        hazards=hazards
    )


def _measurement_evidence_status(
    status: DataStatus,
) -> HazardEvidenceStatus:
    if status == DataStatus.OBSERVED:
        return HazardEvidenceStatus.OBSERVED
    return HazardEvidenceStatus.PREDICTED


def _assess_impact(
    *,
    prediction: RiskPrediction,
    confidence: ConfidenceAssessment,
    landslide: LandslideAssessment,
    cascade: HazardCascadeAssessment,
    roads: tuple[SpatialRoadRiskResult, ...],
    drainage: DrainNetworkResult | None,
    route: RouteDomainResult,
    context: PredictorContext,
) -> WardImpactAssessment | None:
    exposure = context.evacuation.exposure
    if exposure is None:
        return None

    high_risk_roads = sum(
        road.assessment.risk_level
        in {
            RoadFloodRiskLevel.HIGH,
            RoadFloodRiskLevel.SEVERE,
        }
        or road.assessment.recommendation
        in {
            RoadRecommendation.AVOID,
            RoadRecommendation.CLOSED,
        }
        for road in roads
    )

    overflowing_drains = (
        sum(
            drain.overflow_discharge_m3_per_s > 0.0
            or drain.risk.overflow_expected
            for drain in drainage.drain_states
        )
        if drainage is not None
        else 0
    )

    cascade_intensity = max(
        (
            hazard.intensity_score
            for hazard in cascade.inferred_hazards
        ),
        default=0.0,
    )
    cascade_confidence = min(
        (
            hazard.confidence
            for hazard in cascade.inferred_hazards
        ),
        default=confidence.overall_confidence,
    )

    return assess_ward_impact(
        exposure=exposure,
        hazards=WardHazardInputs(
            flood_risk_score=prediction.risk_score,
            flood_confidence=confidence.overall_confidence,
            landslide_risk_score=(
                landslide.susceptibility_score
            ),
            landslide_confidence=landslide.confidence,
            cascade_intensity_score=cascade_intensity,
            cascade_confidence=cascade_confidence,
            high_risk_road_count=high_risk_roads,
            overflowing_drain_count=overflowing_drains,
            route_readiness=_route_readiness(route),
        ),
    )


def _route_readiness(
    route: RouteDomainResult,
) -> RouteReadiness:
    if route.status == RouteStatus.NO_SAFE_ROUTE:
        return RouteReadiness.UNSAFE
    if route.route is None:
        return RouteReadiness.UNKNOWN
    if route.route.safety_level == RouteSafetyLevel.SAFE:
        return RouteReadiness.AVAILABLE
    return RouteReadiness.DEGRADED


def _assess_evacuation_readiness(
    *,
    ward_impact: WardImpactAssessment | None,
    trajectory: RiskTrajectoryAssessment,
    route: RouteDomainResult,
    context: EvacuationContext,
) -> EvacuationReadinessAssessment | None:
    if ward_impact is None:
        return None
    if not context.shelters and not context.routes:
        return None

    route_options = context.routes
    if route.route is not None and not route_options:
        route_options = (
            EvacuationRouteOption(
                route_id="selected_route",
                shelter_id=(
                    context.shelters[0].shelter_id
                    if context.shelters
                    else "unknown_shelter"
                ),
                route_status=(
                    EvacuationRouteStatus.SAFE
                    if route.route.safety_level
                    == RouteSafetyLevel.SAFE
                    else EvacuationRouteStatus.CAUTION
                ),
                travel_time_minutes=(
                    route.route.travel_time_minutes
                ),
                maximum_risk_score=(
                    route.route.maximum_risk_score
                ),
                minimum_confidence=(
                    route.route.minimum_confidence
                ),
            ),
        )

    projection = _best_threshold_projection(trajectory)
    return assess_evacuation_readiness(
        inputs=EvacuationReadinessInputs(
            unit_id=ward_impact.unit_id,
            population_exposed=(
                ward_impact.population_exposed
            ),
            current_impact_score=ward_impact.impact_score,
            current_impact_confidence=ward_impact.confidence,
            projected_threshold_minutes=(
                projection.estimated_minutes_to_crossing
                if projection is not None
                else None
            ),
            projected_threshold_earliest_minutes=(
                projection.earliest_minutes
                if projection is not None
                else None
            ),
            trajectory_confidence=(
                trajectory.trajectory_confidence
            ),
        ),
        shelters=context.shelters,
        routes=route_options,
    )


def _assess_evacuation_clearance(
    *,
    ward_impact: WardImpactAssessment | None,
    trajectory: RiskTrajectoryAssessment,
    context: EvacuationContext,
) -> EvacuationClearanceAssessment | None:
    if (
        ward_impact is None
        or ward_impact.population_exposed is None
        or not context.clearance_paths
    ):
        return None

    projection = _best_threshold_projection(trajectory)
    hazard_window = (
        projection.earliest_minutes
        if projection is not None
        else None
    )
    return assess_evacuation_clearance(
        inputs=EvacuationClearanceInputs(
            unit_id=ward_impact.unit_id,
            population_to_move=(
                ward_impact.population_exposed
            ),
            mobilization_delay_minutes=(
                context.mobilization_delay_minutes
            ),
            hazard_window_minutes=hazard_window,
            data_confidence=ward_impact.confidence,
        ),
        paths=context.clearance_paths,
    )


def _best_threshold_projection(
    trajectory: RiskTrajectoryAssessment,
) -> Any | None:
    projected = [
        projection
        for projection in trajectory.projections
        if (
            projection.status
            == ThresholdStatus.PROJECTED
            and projection.estimated_minutes_to_crossing
            is not None
        )
    ]
    if not projected:
        return None
    return min(
        projected,
        key=lambda projection: (
            projection.estimated_minutes_to_crossing
        ),
    )


def _snapshot_id(
    generated_at: datetime,
) -> str:
    utc = generated_at.astimezone(
        timezone.utc
    )
    return "snap_" + utc.strftime("%Y%m%dT%H%M%SZ")


def _prediction_id(
    catchment_id: str,
    generated_at: datetime,
) -> str:
    utc = generated_at.astimezone(
        timezone.utc
    )
    safe_catchment = catchment_id.replace(
        "_",
        "-",
    )
    return (
        f"PRED-{safe_catchment}-"
        + utc.strftime("%Y%m%dT%H%M%SZ")
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {
            field.name: _plain(
                getattr(value, field.name)
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [
            _plain(item)
            for item in value
        ]
    return value
