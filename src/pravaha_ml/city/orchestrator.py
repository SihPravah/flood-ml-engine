from datetime import datetime
from typing import Iterable

from pravaha_ml.city.models import (
    CatchmentIntelligence,
    CityIntelligenceState,
    CityIntelligenceSummary,
    CityOperationalStatus,
)
from pravaha_ml.drainage.models import (
    DrainRiskLevel,
)
from pravaha_ml.drainage.network import (
    DrainFlowState,
)
from pravaha_ml.inference.confidence import (
    ConfidenceLevel,
    PredictionDisposition,
)
from pravaha_ml.roads.models import (
    RoadFloodRiskLevel,
    RoadRecommendation,
)
from pravaha_ml.roads.spatial import (
    SpatialRoadRiskResult,
)


class CityIntelligenceError(
    ValueError
):
    pass


def _validate_generated_at(
    generated_at: datetime,
) -> None:
    if generated_at.tzinfo is None:
        raise CityIntelligenceError(
            "generated_at must be timezone-aware."
        )


def _validate_unique_ids(
    *,
    catchments: tuple[
        CatchmentIntelligence,
        ...
    ],
    drains: tuple[
        DrainFlowState,
        ...
    ],
    roads: tuple[
        SpatialRoadRiskResult,
        ...
    ],
) -> None:
    catchment_ids = [
        item.catchment_id
        for item in catchments
    ]

    drain_ids = [
        item.drain_id
        for item in drains
    ]

    road_ids = [
        item.road.road_id
        for item in roads
    ]

    if len(catchment_ids) != len(
        set(catchment_ids)
    ):
        raise CityIntelligenceError(
            "Duplicate catchment_id detected."
        )

    if len(drain_ids) != len(
        set(drain_ids)
    ):
        raise CityIntelligenceError(
            "Duplicate drain_id detected."
        )

    if len(road_ids) != len(
        set(road_ids)
    ):
        raise CityIntelligenceError(
            "Duplicate road_id detected."
        )


def _catchment_is_high_risk(
    catchment: CatchmentIntelligence,
) -> bool:
    return (
        catchment.risk_score >= 0.70
        or catchment.risk_level.upper()
        in {
            "HIGH",
            "SEVERE",
        }
    )


def _catchment_has_low_confidence(
    catchment: CatchmentIntelligence,
) -> bool:
    return (
        catchment.confidence.confidence_level
        in {
            ConfidenceLevel.LOW,
            ConfidenceLevel.INSUFFICIENT,
        }
        or catchment.confidence.disposition
        == PredictionDisposition.INSUFFICIENT_DATA
    )


def _drain_is_overflowing(
    drain: DrainFlowState,
) -> bool:
    return (
        drain.overflow_discharge_m3_per_s
        > 0.0
        or drain.risk.overflow_expected
    )


def _road_is_avoid(
    road: SpatialRoadRiskResult,
) -> bool:
    return (
        road.assessment.recommendation
        == RoadRecommendation.AVOID
    )


def _road_is_closed(
    road: SpatialRoadRiskResult,
) -> bool:
    return (
        road.assessment.recommendation
        == RoadRecommendation.CLOSED
    )


def _road_has_low_confidence(
    road: SpatialRoadRiskResult,
) -> bool:
    return (
        road.assessment.data_confidence
        < 0.60
    )


def determine_city_operational_status(
    *,
    catchments: Iterable[
        CatchmentIntelligence
    ],
    drains: Iterable[
        DrainFlowState
    ],
    roads: Iterable[
        SpatialRoadRiskResult
    ],
) -> CityOperationalStatus:
    """
    Derive high-level city operational state.

    Priority:

        INSUFFICIENT_DATA
            when evidence quality is too weak across a large
            fraction of the monitored system.

        EMERGENCY
            when severe/high flood indicators or confirmed road
            closures are present.

        ELEVATED
            when warning-level conditions, overflow, or road
            avoidance exists.

        NORMAL
            otherwise.

    These rules are development decision-support policy and must
    later be calibrated with disaster-management stakeholders.
    """

    catchments = tuple(
        catchments
    )

    drains = tuple(
        drains
    )

    roads = tuple(
        roads
    )

    total_monitored = (
        len(catchments)
        + len(roads)
    )

    low_confidence_items = (
        sum(
            _catchment_has_low_confidence(
                catchment
            )
            for catchment in catchments
        )
        + sum(
            _road_has_low_confidence(
                road
            )
            for road in roads
        )
    )

    if (
        total_monitored > 0
        and (
            low_confidence_items
            / total_monitored
        ) >= 0.50
    ):
        return (
            CityOperationalStatus
            .INSUFFICIENT_DATA
        )

    severe_catchment = any(
        catchment.risk_level.upper()
        == "SEVERE"
        or catchment.risk_score >= 0.85
        for catchment in catchments
    )

    severe_drain = any(
        drain.risk.risk_level
        == DrainRiskLevel.SEVERE
        for drain in drains
    )

    severe_road = any(
        road.assessment.risk_level
        == RoadFloodRiskLevel.SEVERE
        for road in roads
    )

    confirmed_closure = any(
        _road_is_closed(
            road
        )
        for road in roads
    )

    if (
        severe_catchment
        or severe_drain
        or severe_road
        or confirmed_closure
    ):
        return (
            CityOperationalStatus.EMERGENCY
        )

    elevated_catchment = any(
        _catchment_is_high_risk(
            catchment
        )
        for catchment in catchments
    )

    overflowing_drain = any(
        _drain_is_overflowing(
            drain
        )
        for drain in drains
    )

    unsafe_road = any(
        _road_is_avoid(
            road
        )
        for road in roads
    )

    warning_road = any(
        road.assessment.risk_level
        in {
            RoadFloodRiskLevel.WARNING,
            RoadFloodRiskLevel.HIGH,
        }
        for road in roads
    )

    if (
        elevated_catchment
        or overflowing_drain
        or unsafe_road
        or warning_road
    ):
        return (
            CityOperationalStatus.ELEVATED
        )

    return (
        CityOperationalStatus.NORMAL
    )


def build_city_intelligence_state(
    *,
    generated_at: datetime,
    catchments: Iterable[
        CatchmentIntelligence
    ],
    drains: Iterable[
        DrainFlowState
    ],
    roads: Iterable[
        SpatialRoadRiskResult
    ],
) -> CityIntelligenceState:
    """
    Consolidate the current PRAVAHA analysis into one city-level
    intelligence state.

    This intentionally contains detailed objects rather than
    flattening everything into presentation JSON.

    A future backend adapter can translate this internal state
    into whatever shared frontend/API contract the team agrees.
    """

    _validate_generated_at(
        generated_at
    )

    catchments = tuple(
        catchments
    )

    drains = tuple(
        drains
    )

    roads = tuple(
        roads
    )

    _validate_unique_ids(
        catchments=catchments,
        drains=drains,
        roads=roads,
    )

    high_risk_catchments = sum(
        _catchment_is_high_risk(
            catchment
        )
        for catchment in catchments
    )

    overflowing_drains = sum(
        _drain_is_overflowing(
            drain
        )
        for drain in drains
    )

    roads_to_avoid = sum(
        _road_is_avoid(
            road
        )
        for road in roads
    )

    confirmed_road_closures = sum(
        _road_is_closed(
            road
        )
        for road in roads
    )

    low_confidence_catchments = sum(
        _catchment_has_low_confidence(
            catchment
        )
        for catchment in catchments
    )

    low_confidence_roads = sum(
        _road_has_low_confidence(
            road
        )
        for road in roads
    )

    operational_status = (
        determine_city_operational_status(
            catchments=catchments,
            drains=drains,
            roads=roads,
        )
    )

    summary = CityIntelligenceSummary(
        generated_at=generated_at,
        catchment_count=len(
            catchments
        ),
        drain_count=len(
            drains
        ),
        road_count=len(
            roads
        ),
        high_risk_catchments=int(
            high_risk_catchments
        ),
        overflowing_drains=int(
            overflowing_drains
        ),
        roads_to_avoid=int(
            roads_to_avoid
        ),
        confirmed_road_closures=int(
            confirmed_road_closures
        ),
        low_confidence_catchments=int(
            low_confidence_catchments
        ),
        low_confidence_roads=int(
            low_confidence_roads
        ),
        operational_status=(
            operational_status
        ),
    )

    return CityIntelligenceState(
        generated_at=generated_at,
        catchments=catchments,
        drains=drains,
        roads=roads,
        summary=summary,
    )


def get_catchment_intelligence(
    *,
    state: CityIntelligenceState,
    catchment_id: str,
) -> CatchmentIntelligence:
    for catchment in state.catchments:
        if (
            catchment.catchment_id
            == catchment_id
        ):
            return catchment

    raise LookupError(
        f"Catchment not found: {catchment_id}"
    )


def get_drain_intelligence(
    *,
    state: CityIntelligenceState,
    drain_id: str,
) -> DrainFlowState:
    for drain in state.drains:
        if drain.drain_id == drain_id:
            return drain

    raise LookupError(
        f"Drain not found: {drain_id}"
    )


def get_road_intelligence(
    *,
    state: CityIntelligenceState,
    road_id: str,
) -> SpatialRoadRiskResult:
    for road in state.roads:
        if road.road.road_id == road_id:
            return road

    raise LookupError(
        f"Road not found: {road_id}"
    )