import pytest

from shapely.geometry import LineString

from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainStaticProfile,
)
from pravaha_ml.drainage.network import (
    route_drainage_network,
)
from pravaha_ml.drainage.flow import (
    CatchmentRunoffInput,
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
    DrainAssociationError,
    RoadEnvironmentalContext,
    assess_road_from_spatial_context,
    build_road_flood_inputs_from_spatial_context,
    find_nearest_drain,
)


def make_road() -> RoadSegment:
    return RoadSegment(
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


def make_drain(
    drain_id: str,
    latitude_offset: float,
) -> DrainageSegment:
    return DrainageSegment(
        drain_id=drain_id,
        geometry=LineString(
            [
                (
                    77.0000,
                    30.0000
                    + latitude_offset,
                ),
                (
                    77.0020,
                    30.0000
                    + latitude_offset,
                ),
            ]
        ),
        provenance=(
            SpatialDataProvenance.VERIFIED
        ),
        catchment_id="C_001",
    )


def make_static_profile(
    drain_id: str,
) -> DrainStaticProfile:
    return DrainStaticProfile(
        drain_id=drain_id,
        width_m=0.30,
        depth_m=0.30,
        slope_fraction=0.01,
        manning_roughness=0.015,
        blockage_fraction=0.20,
        condition=(
            DrainCondition.DEGRADED
        ),
        capacity_provenance=(
            DrainCapacityProvenance.VERIFIED
        ),
        catchment_id="C_001",
    )


def make_context(
    *,
    confidence: float = 0.90,
) -> RoadEnvironmentalContext:
    return RoadEnvironmentalContext(
        catchment_risk_score=0.80,
        terrain_depression_score=0.80,
        stream_proximity_score=0.20,
        historical_waterlogging_score=0.70,
        road_surface_vulnerability_score=0.50,
        data_confidence=confidence,
    )


def make_flow_states():
    return route_drainage_network(
        profiles=[
            make_static_profile(
                "D_NEAR"
            ),
            make_static_profile(
                "D_FAR"
            ),
        ],
        connections=[],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_001",
                drain_id="D_NEAR",
                catchment_area_km2=0.50,
                runoff_mm=30.0,
                response_time_minutes=20.0,
                data_confidence=0.80,
            ),
            CatchmentRunoffInput(
                catchment_id="C_002",
                drain_id="D_FAR",
                catchment_area_km2=0.05,
                runoff_mm=2.0,
                response_time_minutes=60.0,
                data_confidence=0.95,
            ),
        ],
    ).drain_states


def test_nearest_drain_is_selected():
    road = make_road()

    near = make_drain(
        "D_NEAR",
        0.0001,
    )

    far = make_drain(
        "D_FAR",
        0.0100,
    )

    result = find_nearest_drain(
        road=road,
        drains=[
            far,
            near,
        ],
    )

    assert (
        result.drain_id
        == "D_NEAR"
    )

    assert result.distance_m > 0.0


def test_association_distance_is_metric():
    result = find_nearest_drain(
        road=make_road(),
        drains=[
            make_drain(
                "D_NEAR",
                0.0010,
            )
        ],
    )

    assert result.distance_m > 50.0
    assert result.distance_m < 150.0


def test_maximum_distance_rejects_far_drain():
    with pytest.raises(
        DrainAssociationError,
        match=(
            "No drain lies within"
        ),
    ):
        find_nearest_drain(
            road=make_road(),
            drains=[
                make_drain(
                    "D_FAR",
                    0.0100,
                )
            ],
            maximum_distance_m=50.0,
        )


def test_empty_drain_collection_rejected():
    with pytest.raises(
        DrainAssociationError,
        match=(
            "No drainage segments are available"
        ),
    ):
        find_nearest_drain(
            road=make_road(),
            drains=[],
        )


def test_invalid_maximum_distance_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "maximum_distance_m must be greater than 0"
        ),
    ):
        find_nearest_drain(
            road=make_road(),
            drains=[
                make_drain(
                    "D_NEAR",
                    0.0001,
                )
            ],
            maximum_distance_m=0.0,
        )


def test_spatial_builder_uses_matching_flow_state():
    road = make_road()

    drains = [
        make_drain(
            "D_NEAR",
            0.0001,
        ),
        make_drain(
            "D_FAR",
            0.0100,
        ),
    ]

    (
        inputs,
        association,
    ) = build_road_flood_inputs_from_spatial_context(
        road=road,
        drains=drains,
        drain_flow_states=(
            make_flow_states()
        ),
        context=make_context(),
    )

    assert (
        association.drain_id
        == "D_NEAR"
    )

    assert (
        inputs.road_id
        == "R_001"
    )

    assert (
        inputs.drain_capacity_utilization
        > 0.0
    )


def test_combined_confidence_uses_weaker_source():
    (
        inputs,
        _,
    ) = build_road_flood_inputs_from_spatial_context(
        road=make_road(),
        drains=[
            make_drain(
                "D_NEAR",
                0.0001,
            )
        ],
        drain_flow_states=(
            make_flow_states()
        ),
        context=make_context(
            confidence=0.95
        ),
    )

    assert (
        inputs.data_confidence
        == pytest.approx(0.80)
    )


def test_missing_dynamic_state_is_not_silently_accepted():
    with pytest.raises(
        DrainAssociationError,
        match=(
            "No drainage segment has a matching "
            "dynamic flow state"
        ),
    ):
        build_road_flood_inputs_from_spatial_context(
            road=make_road(),
            drains=[
                make_drain(
                    "D_UNKNOWN",
                    0.0001,
                )
            ],
            drain_flow_states=(
                make_flow_states()
            ),
            context=make_context(),
        )


def test_end_to_end_spatial_road_assessment():
    result = (
        assess_road_from_spatial_context(
            road=make_road(),
            drains=[
                make_drain(
                    "D_NEAR",
                    0.0001,
                ),
                make_drain(
                    "D_FAR",
                    0.0100,
                ),
            ],
            drain_flow_states=(
                make_flow_states()
            ),
            context=make_context(),
        )
    )

    assert (
        result.association.drain_id
        == "D_NEAR"
    )

    assert (
        result.assessment.road_id
        == "R_001"
    )

    assert (
        result.assessment.risk_score
        > 0.0
    )


def test_authority_closure_propagates_through_spatial_builder():
    context = RoadEnvironmentalContext(
        catchment_risk_score=0.10,
        terrain_depression_score=0.10,
        stream_proximity_score=0.10,
        historical_waterlogging_score=0.10,
        road_surface_vulnerability_score=0.10,
        data_confidence=0.90,
        authority_closed=True,
    )

    result = (
        assess_road_from_spatial_context(
            road=make_road(),
            drains=[
                make_drain(
                    "D_NEAR",
                    0.0001,
                )
            ],
            drain_flow_states=(
                make_flow_states()
            ),
            context=context,
        )
    )

    assert (
        result.assessment.recommendation
        == RoadRecommendation.CLOSED
    )


def test_invalid_environmental_context_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "terrain_depression_score must be between 0 and 1"
        ),
    ):
        RoadEnvironmentalContext(
            catchment_risk_score=0.50,
            terrain_depression_score=1.50,
            stream_proximity_score=0.20,
            historical_waterlogging_score=0.20,
            road_surface_vulnerability_score=0.20,
            data_confidence=0.90,
        )


def test_nearest_geometric_drain_without_state_is_skipped():
    road = make_road()

    no_state_drain = make_drain(
        "D_NO_STATE",
        0.00005,
    )

    valid_drain = make_drain(
        "D_NEAR",
        0.0005,
    )

    (
        _,
        association,
    ) = build_road_flood_inputs_from_spatial_context(
        road=road,
        drains=[
            no_state_drain,
            valid_drain,
        ],
        drain_flow_states=(
            make_flow_states()
        ),
        context=make_context(),
    )

    assert (
        association.drain_id
        == "D_NEAR"
    )