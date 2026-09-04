from dataclasses import dataclass
from typing import Iterable

from pravaha_ml.drainage.network import (
    DrainFlowState,
)
from pravaha_ml.geospatial.metric import (
    line_distance_metres,
)
from pravaha_ml.geospatial.models import (
    DrainageSegment,
    RoadSegment,
)
from pravaha_ml.roads.models import (
    RoadFloodAssessment,
    RoadFloodInputs,
)
from pravaha_ml.roads.risk import (
    assess_road_flood_risk,
)


@dataclass(frozen=True)
class RoadEnvironmentalContext:
    """
    Non-drainage environmental evidence required by the road-risk
    engine.

    These values will eventually come from GIS/history/model
    services.

    Keeping them explicit prevents the spatial association layer
    from silently inventing missing evidence.
    """

    catchment_risk_score: float
    terrain_depression_score: float
    stream_proximity_score: float
    historical_waterlogging_score: float
    road_surface_vulnerability_score: float

    data_confidence: float

    authority_closed: bool = False

    def __post_init__(self) -> None:
        normalized = {
            "catchment_risk_score": (
                self.catchment_risk_score
            ),
            "terrain_depression_score": (
                self.terrain_depression_score
            ),
            "stream_proximity_score": (
                self.stream_proximity_score
            ),
            "historical_waterlogging_score": (
                self.historical_waterlogging_score
            ),
            "road_surface_vulnerability_score": (
                self.road_surface_vulnerability_score
            ),
            "data_confidence": (
                self.data_confidence
            ),
        }

        for field_name, value in normalized.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )


@dataclass(frozen=True)
class RoadDrainAssociation:
    road_id: str
    drain_id: str

    distance_m: float

    projected_crs_epsg: int


@dataclass(frozen=True)
class SpatialRoadRiskResult:
    road: RoadSegment

    association: RoadDrainAssociation

    road_inputs: RoadFloodInputs

    assessment: RoadFloodAssessment


class DrainAssociationError(
    LookupError
):
    pass


def _build_flow_state_map(
    states: Iterable[
        DrainFlowState
    ],
) -> dict[
    str,
    DrainFlowState,
]:
    states = list(
        states
    )

    state_map: dict[
        str,
        DrainFlowState,
    ] = {}

    for state in states:
        if state.drain_id in state_map:
            raise ValueError(
                f"Duplicate drain flow state: {state.drain_id}"
            )

        state_map[
            state.drain_id
        ] = state

    return state_map


def find_nearest_drain(
    *,
    road: RoadSegment,
    drains: Iterable[
        DrainageSegment
    ],
    maximum_distance_m: float | None = None,
) -> RoadDrainAssociation:
    """
    Find the nearest mapped drainage segment to a road.

    Distance is calculated in metres using local projected
    coordinates.

    maximum_distance_m:
        Optional spatial influence cutoff.

        When supplied, a drain farther away than this threshold is
        not silently treated as relevant.
    """

    drains = list(
        drains
    )

    if not drains:
        raise DrainAssociationError(
            "No drainage segments are available."
        )

    if (
        maximum_distance_m is not None
        and maximum_distance_m <= 0.0
    ):
        raise ValueError(
            "maximum_distance_m must be greater than 0."
        )

    best_association: (
        RoadDrainAssociation
        | None
    ) = None

    for drain in drains:
        distance = line_distance_metres(
            road.geometry,
            drain.geometry,
        )

        candidate = RoadDrainAssociation(
            road_id=road.road_id,
            drain_id=drain.drain_id,
            distance_m=(
                distance.distance_m
            ),
            projected_crs_epsg=(
                distance.projected_crs_epsg
            ),
        )

        if (
            best_association is None
            or candidate.distance_m
            < best_association.distance_m
        ):
            best_association = candidate

    if best_association is None:
        raise DrainAssociationError(
            "Unable to associate road with a drain."
        )

    if (
        maximum_distance_m is not None
        and best_association.distance_m
        > maximum_distance_m
    ):
        raise DrainAssociationError(
            "No drain lies within the configured "
            "maximum association distance."
        )

    return best_association


def build_road_flood_inputs_from_spatial_context(
    *,
    road: RoadSegment,
    drains: Iterable[
        DrainageSegment
    ],
    drain_flow_states: Iterable[
        DrainFlowState
    ],
    context: RoadEnvironmentalContext,
    maximum_drain_distance_m: (
        float | None
    ) = None,
) -> tuple[
    RoadFloodInputs,
    RoadDrainAssociation,
]:
    """
    Automatically connect a road to the nearest relevant drain
    and construct RoadFloodInputs from the associated dynamic
    drainage state.

    This is the bridge:

        road geometry
            +
        drain geometry
            +
        drainage-network state
            ↓
        RoadFloodInputs

    The nearest mapped drain must also have a corresponding
    dynamic DrainFlowState.
    """

    drains = list(
        drains
    )

    flow_state_map = (
        _build_flow_state_map(
            drain_flow_states
        )
    )

    eligible_drains = [
        drain
        for drain in drains
        if drain.drain_id
        in flow_state_map
    ]

    if not eligible_drains:
        raise DrainAssociationError(
            "No drainage segment has a matching "
            "dynamic flow state."
        )

    association = find_nearest_drain(
        road=road,
        drains=eligible_drains,
        maximum_distance_m=(
            maximum_drain_distance_m
        ),
    )

    state = flow_state_map[
        association.drain_id
    ]

    combined_confidence = min(
        context.data_confidence,
        state.data_confidence,
    )

    inputs = RoadFloodInputs(
        road_id=road.road_id,
        catchment_risk_score=(
            context.catchment_risk_score
        ),
        drain_overflow_probability=(
            state.risk.overflow_probability
        ),
        drain_capacity_utilization=(
            state.capacity_utilization
        ),
        drain_overflow_discharge_m3_per_s=(
            state.overflow_discharge_m3_per_s
        ),
        nearest_drain_distance_m=(
            association.distance_m
        ),
        terrain_depression_score=(
            context.terrain_depression_score
        ),
        stream_proximity_score=(
            context.stream_proximity_score
        ),
        historical_waterlogging_score=(
            context.historical_waterlogging_score
        ),
        road_surface_vulnerability_score=(
            context
            .road_surface_vulnerability_score
        ),
        data_confidence=(
            combined_confidence
        ),
        authority_closed=(
            context.authority_closed
        ),
    )

    return (
        inputs,
        association,
    )


def assess_road_from_spatial_context(
    *,
    road: RoadSegment,
    drains: Iterable[
        DrainageSegment
    ],
    drain_flow_states: Iterable[
        DrainFlowState
    ],
    context: RoadEnvironmentalContext,
    maximum_drain_distance_m: (
        float | None
    ) = None,
) -> SpatialRoadRiskResult:
    """
    End-to-end spatial road-risk assembly.

    This function automatically:

        finds the relevant nearest drain
        retrieves its live/network state
        constructs RoadFloodInputs
        evaluates road flood risk
    """

    (
        inputs,
        association,
    ) = build_road_flood_inputs_from_spatial_context(
        road=road,
        drains=drains,
        drain_flow_states=drain_flow_states,
        context=context,
        maximum_drain_distance_m=(
            maximum_drain_distance_m
        ),
    )

    assessment = assess_road_flood_risk(
        inputs
    )

    return SpatialRoadRiskResult(
        road=road,
        association=association,
        road_inputs=inputs,
        assessment=assessment,
    )