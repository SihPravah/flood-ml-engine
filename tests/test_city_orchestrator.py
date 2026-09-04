from datetime import (
    datetime,
    timezone,
)

import pytest
from shapely.geometry import LineString

from pravaha_ml.city.models import (
    CatchmentIntelligence,
    CityOperationalStatus,
)
from pravaha_ml.city.orchestrator import (
    CityIntelligenceError,
    build_city_intelligence_state,
    determine_city_operational_status,
    get_catchment_intelligence,
    get_drain_intelligence,
    get_road_intelligence,
)
from pravaha_ml.drainage.flow import (
    CatchmentRunoffInput,
)
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainStaticProfile,
)
from pravaha_ml.drainage.network import (
    route_drainage_network,
)
from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.inference.confidence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    PredictionDisposition,
)
from pravaha_ml.geospatial.models import (
    DrainageSegment,
    RoadSegment,
    SpatialDataProvenance,
)
from pravaha_ml.roads.models import (
    RoadRecommendation,
)
from pravaha_ml.roads.spatial import (
    RoadEnvironmentalContext,
    assess_road_from_spatial_context,
)


UTC = timezone.utc


def make_hydrology() -> HydrologyFeatures:
    return HydrologyFeatures(
        rain_15m_mm=10.0,
        rain_30m_mm=20.0,
        rain_1h_mm=30.0,
        rain_3h_mm=40.0,
        rain_6h_mm=50.0,
        rain_24h_mm=80.0,
        api_mm=35.0,
        soil_moisture_percentage=80.0,
        soil_saturation=0.80,
        moisture_condition="WET",
        base_curve_number=80.0,
        effective_curve_number=90.0,
        runoff_mm=18.0,
        runoff_ratio=0.60,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        concentration_time_minutes=25.0,
    )


def make_confidence(
    *,
    high: bool = True,
) -> ConfidenceAssessment:
    if high:
        return ConfidenceAssessment(
            overall_confidence=0.90,
            confidence_level=(
                ConfidenceLevel.HIGH
            ),
            disposition=(
                PredictionDisposition.NORMAL
            ),
            temporal_quality_score=1.0,
            source_availability_score=1.0,
            provenance_score=1.0,
            model_agreement_score=0.95,
            estimated_input_fraction=0.0,
            source_availability_fraction=1.0,
            model_disagreement=0.05,
            can_treat_low_risk_as_reliable=True,
            reasons=(),
        )

    return ConfidenceAssessment(
        overall_confidence=0.25,
        confidence_level=(
            ConfidenceLevel.INSUFFICIENT
        ),
        disposition=(
            PredictionDisposition.INSUFFICIENT_DATA
        ),
        temporal_quality_score=0.0,
        source_availability_score=0.30,
        provenance_score=0.30,
        model_agreement_score=0.50,
        estimated_input_fraction=0.50,
        source_availability_fraction=0.30,
        model_disagreement=0.50,
        can_treat_low_risk_as_reliable=False,
        reasons=(
            "insufficient_overall_confidence",
        ),
    )


def make_catchment(
    *,
    catchment_id: str = "C_001",
    risk_score: float = 0.20,
    risk_level: str = "LOW",
    high_confidence: bool = True,
) -> CatchmentIntelligence:
    return CatchmentIntelligence(
        catchment_id=catchment_id,
        risk_score=risk_score,
        risk_level=risk_level,
        confidence=make_confidence(
            high=high_confidence
        ),
        hydrology=make_hydrology(),
    )


def make_drain_state(
    *,
    high_runoff: bool = False,
):
    profile = DrainStaticProfile(
        drain_id="D_001",
        width_m=0.30,
        depth_m=0.30,
        slope_fraction=0.01,
        manning_roughness=0.015,
        blockage_fraction=0.10,
        condition=DrainCondition.GOOD,
        capacity_provenance=(
            DrainCapacityProvenance.VERIFIED
        ),
        catchment_id="C_001",
    )

    runoff_mm = (
        50.0
        if high_runoff
        else 0.5
    )

    result = route_drainage_network(
        profiles=[
            profile
        ],
        connections=[],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_001",
                drain_id="D_001",
                catchment_area_km2=0.50,
                runoff_mm=runoff_mm,
                response_time_minutes=20.0,
                data_confidence=0.90,
            )
        ],
    )

    return result.drain_states[
        0
    ]


def make_spatial_road(
    *,
    authority_closed: bool = False,
    high_risk: bool = False,
    confidence: float = 0.90,
):
    road = RoadSegment(
        road_id="R_001",
        geometry=LineString(
            [
                (77.0000, 30.0000),
                (77.0020, 30.0000),
            ]
        ),
        road_name="Test Road",
        provenance=(
            SpatialDataProvenance.VERIFIED
        ),
        catchment_id="C_001",
    )

    drain = DrainageSegment(
        drain_id="D_001",
        geometry=LineString(
            [
                (77.0000, 30.0001),
                (77.0020, 30.0001),
            ]
        ),
        provenance=(
            SpatialDataProvenance.VERIFIED
        ),
        catchment_id="C_001",
    )

    drain_state = make_drain_state(
        high_runoff=high_risk
    )

    context = RoadEnvironmentalContext(
        catchment_risk_score=(
            0.95
            if high_risk
            else 0.10
        ),
        terrain_depression_score=(
            0.95
            if high_risk
            else 0.10
        ),
        stream_proximity_score=(
            0.90
            if high_risk
            else 0.10
        ),
        historical_waterlogging_score=(
            0.90
            if high_risk
            else 0.10
        ),
        road_surface_vulnerability_score=(
            0.90
            if high_risk
            else 0.10
        ),
        data_confidence=confidence,
        authority_closed=authority_closed,
    )

    return assess_road_from_spatial_context(
        road=road,
        drains=[
            drain
        ],
        drain_flow_states=[
            drain_state
        ],
        context=context,
    )


def test_normal_city_state():
    catchment = make_catchment()

    drain = make_drain_state(
        high_runoff=False
    )

    road = make_spatial_road(
        high_risk=False
    )

    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[
            catchment
        ],
        drains=[
            drain
        ],
        roads=[
            road
        ],
    )

    assert (
        state.summary.catchment_count
        == 1
    )

    assert (
        state.summary.drain_count
        == 1
    )

    assert (
        state.summary.road_count
        == 1
    )


def test_high_risk_catchment_is_counted():
    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[
            make_catchment(
                risk_score=0.80,
                risk_level="HIGH",
            )
        ],
        drains=[],
        roads=[],
    )

    assert (
        state.summary.high_risk_catchments
        == 1
    )


def test_overflowing_drain_is_counted():
    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[],
        drains=[
            make_drain_state(
                high_runoff=True
            )
        ],
        roads=[],
    )

    assert (
        state.summary.overflowing_drains
        == 1
    )


def test_avoid_road_is_counted():
    road = make_spatial_road(
        high_risk=True
    )

    assert (
        road.assessment.recommendation
        == RoadRecommendation.AVOID
    )

    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[],
        drains=[],
        roads=[
            road
        ],
    )

    assert (
        state.summary.roads_to_avoid
        == 1
    )


def test_authority_closure_is_counted():
    road = make_spatial_road(
        authority_closed=True
    )

    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[],
        drains=[],
        roads=[
            road
        ],
    )

    assert (
        state.summary.confirmed_road_closures
        == 1
    )


def test_severe_catchment_causes_emergency():
    status = determine_city_operational_status(
        catchments=[
            make_catchment(
                risk_score=0.90,
                risk_level="SEVERE",
            )
        ],
        drains=[],
        roads=[],
    )

    assert (
        status
        == CityOperationalStatus.EMERGENCY
    )


def test_confirmed_closure_causes_emergency():
    status = determine_city_operational_status(
        catchments=[],
        drains=[],
        roads=[
            make_spatial_road(
                authority_closed=True
            )
        ],
    )

    assert (
        status
        == CityOperationalStatus.EMERGENCY
    )


def test_overflow_can_make_city_elevated():
    status = determine_city_operational_status(
        catchments=[],
        drains=[
            make_drain_state(
                high_runoff=True
            )
        ],
        roads=[],
    )

    assert (
        status
        in {
            CityOperationalStatus.ELEVATED,
            CityOperationalStatus.EMERGENCY,
        }
    )


def test_widespread_low_confidence_causes_insufficient_data():
    status = determine_city_operational_status(
        catchments=[
            make_catchment(
                high_confidence=False
            ),
            make_catchment(
                catchment_id="C_002",
                high_confidence=False,
            ),
        ],
        drains=[],
        roads=[],
    )

    assert (
        status
        == CityOperationalStatus.INSUFFICIENT_DATA
    )


def test_low_confidence_is_counted():
    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[
            make_catchment(
                high_confidence=False
            )
        ],
        drains=[],
        roads=[],
    )

    assert (
        state.summary.low_confidence_catchments
        == 1
    )


def test_naive_generated_at_rejected():
    with pytest.raises(
        CityIntelligenceError,
        match=(
            "generated_at must be timezone-aware"
        ),
    ):
        build_city_intelligence_state(
            generated_at=datetime(
                2026,
                9,
                4,
                12,
                0,
            ),
            catchments=[],
            drains=[],
            roads=[],
        )


def test_duplicate_catchment_ids_rejected():
    with pytest.raises(
        CityIntelligenceError,
        match=(
            "Duplicate catchment_id"
        ),
    ):
        build_city_intelligence_state(
            generated_at=datetime.now(
                UTC
            ),
            catchments=[
                make_catchment(
                    catchment_id="C_001"
                ),
                make_catchment(
                    catchment_id="C_001"
                ),
            ],
            drains=[],
            roads=[],
        )


def test_catchment_lookup():
    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[
            make_catchment(
                catchment_id="C_123"
            )
        ],
        drains=[],
        roads=[],
    )

    result = get_catchment_intelligence(
        state=state,
        catchment_id="C_123",
    )

    assert (
        result.catchment_id
        == "C_123"
    )


def test_drain_lookup():
    drain = make_drain_state()

    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[],
        drains=[
            drain
        ],
        roads=[],
    )

    result = get_drain_intelligence(
        state=state,
        drain_id="D_001",
    )

    assert result.drain_id == "D_001"


def test_road_lookup():
    road = make_spatial_road()

    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[],
        drains=[],
        roads=[
            road
        ],
    )

    result = get_road_intelligence(
        state=state,
        road_id="R_001",
    )

    assert (
        result.road.road_id
        == "R_001"
    )


def test_unknown_road_lookup_rejected():
    state = build_city_intelligence_state(
        generated_at=datetime.now(
            UTC
        ),
        catchments=[],
        drains=[],
        roads=[],
    )

    with pytest.raises(
        LookupError,
        match="Road not found",
    ):
        get_road_intelligence(
            state=state,
            road_id="UNKNOWN",
        )