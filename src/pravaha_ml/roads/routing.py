from dataclasses import dataclass
from enum import Enum
import heapq
from math import isfinite
from typing import Iterable

from pravaha_ml.roads.models import (
    RoadFloodAssessment,
    RoadFloodRiskLevel,
    RoadRecommendation,
)


class RouteSafetyLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH_RISK = "HIGH_RISK"


@dataclass(frozen=True)
class RoadGraphEdge:
    """
    One directed edge in PRAVAHA's routable road graph.

    start_node / end_node:
        Internal graph node identifiers.

    road_id:
        Identifier linking this routing edge to the road-segment
        intelligence layer.

    travel_time_minutes:
        Estimated traversal time for this edge.

    assessment:
        Current flood-risk assessment for the road segment.
    """

    start_node: str
    end_node: str

    road_id: str

    travel_time_minutes: float

    assessment: RoadFloodAssessment

    def __post_init__(self) -> None:
        if not self.start_node.strip():
            raise ValueError(
                "start_node cannot be empty."
            )

        if not self.end_node.strip():
            raise ValueError(
                "end_node cannot be empty."
            )

        if self.start_node == self.end_node:
            raise ValueError(
                "Road graph edge cannot connect a node to itself."
            )

        if not self.road_id.strip():
            raise ValueError(
                "road_id cannot be empty."
            )

        if self.travel_time_minutes <= 0.0:
            raise ValueError(
                "travel_time_minutes must be greater than 0."
            )

        if not isfinite(
            self.travel_time_minutes
        ):
            raise ValueError(
                "travel_time_minutes must be finite."
            )

        if (
            self.assessment.road_id
            != self.road_id
        ):
            raise ValueError(
                "road_id must match assessment.road_id."
            )


@dataclass(frozen=True)
class RoutingPolicy:
    """
    Development policy controlling how strongly PRAVAHA trades
    travel time against flood risk and uncertainty.

    risk_penalty_minutes:
        Maximum additional edge penalty produced by risk_score.

    uncertainty_penalty_minutes:
        Maximum additional penalty produced by low confidence.

    watch_penalty_minutes / warning_penalty_minutes:
        Additional categorical penalties.

    avoid_penalty_minutes:
        Used only if allow_avoid_segments=True.

    allow_avoid_segments:
        False by default. Predicted AVOID segments are excluded
        from normal routing.

    closed segments are ALWAYS excluded.
    """

    risk_penalty_minutes: float = 20.0
    uncertainty_penalty_minutes: float = 12.0

    watch_penalty_minutes: float = 2.0
    warning_penalty_minutes: float = 8.0
    high_penalty_minutes: float = 20.0
    severe_penalty_minutes: float = 40.0

    avoid_penalty_minutes: float = 60.0

    allow_avoid_segments: bool = False


@dataclass(frozen=True)
class RouteSegmentDecision:
    road_id: str
    start_node: str
    end_node: str

    travel_time_minutes: float
    routing_cost: float

    risk_score: float
    risk_level: RoadFloodRiskLevel
    recommendation: RoadRecommendation
    confidence: float

    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SafeRouteResult:
    start_node: str
    destination_node: str

    node_path: tuple[str, ...]
    segments: tuple[
        RouteSegmentDecision,
        ...
    ]

    travel_time_minutes: float
    routing_cost: float

    maximum_risk_score: float
    minimum_confidence: float

    safety_level: RouteSafetyLevel

    avoided_closed_segments: int
    avoided_predicted_unsafe_segments: int

    reasons: tuple[str, ...]


class RouteNotFoundError(
    LookupError
):
    pass


def _validate_policy(
    policy: RoutingPolicy,
) -> None:
    penalty_values = [
        policy.risk_penalty_minutes,
        policy.uncertainty_penalty_minutes,
        policy.watch_penalty_minutes,
        policy.warning_penalty_minutes,
        policy.high_penalty_minutes,
        policy.severe_penalty_minutes,
        policy.avoid_penalty_minutes,
    ]

    if any(
        value < 0.0
        for value in penalty_values
    ):
        raise ValueError(
            "Routing penalties cannot be negative."
        )


def _risk_level_penalty(
    risk_level: RoadFloodRiskLevel,
    policy: RoutingPolicy,
) -> float:
    if risk_level == RoadFloodRiskLevel.LOW:
        return 0.0

    if risk_level == RoadFloodRiskLevel.WATCH:
        return policy.watch_penalty_minutes

    if risk_level == RoadFloodRiskLevel.WARNING:
        return policy.warning_penalty_minutes

    if risk_level == RoadFloodRiskLevel.HIGH:
        return policy.high_penalty_minutes

    return policy.severe_penalty_minutes


def edge_is_routable(
    edge: RoadGraphEdge,
    policy: RoutingPolicy,
) -> bool:
    """
    Determine whether an edge may participate in route search.

    Authority-confirmed CLOSED:
        always excluded.

    PRAVAHA-predicted AVOID:
        excluded by default, but may be admitted when
        allow_avoid_segments=True for fallback/emergency analysis.
    """

    recommendation = (
        edge.assessment.recommendation
    )

    if (
        recommendation
        == RoadRecommendation.CLOSED
    ):
        return False

    if (
        recommendation
        == RoadRecommendation.AVOID
        and not policy.allow_avoid_segments
    ):
        return False

    return True


def calculate_edge_routing_cost(
    edge: RoadGraphEdge,
    policy: RoutingPolicy | None = None,
) -> float:
    """
    Calculate generalized routing cost for one road segment.

    This cost is NOT travel time.

    It combines:

        actual travel time
        + continuous flood-risk penalty
        + confidence/uncertainty penalty
        + categorical risk penalty
        + optional AVOID penalty

    This allows PRAVAHA to prefer a slightly longer but
    substantially safer route.
    """

    if policy is None:
        policy = RoutingPolicy()

    _validate_policy(
        policy
    )

    if not edge_is_routable(
        edge,
        policy,
    ):
        return float(
            "inf"
        )

    assessment = (
        edge.assessment
    )

    risk_penalty = (
        assessment.risk_score
        * policy.risk_penalty_minutes
    )

    uncertainty_penalty = (
        1.0
        - assessment.data_confidence
    ) * policy.uncertainty_penalty_minutes

    categorical_penalty = (
        _risk_level_penalty(
            assessment.risk_level,
            policy,
        )
    )

    avoid_penalty = 0.0

    if (
        assessment.recommendation
        == RoadRecommendation.AVOID
    ):
        avoid_penalty = (
            policy.avoid_penalty_minutes
        )

    return float(
        edge.travel_time_minutes
        + risk_penalty
        + uncertainty_penalty
        + categorical_penalty
        + avoid_penalty
    )


def _build_adjacency(
    edges: Iterable[
        RoadGraphEdge
    ],
) -> dict[
    str,
    list[RoadGraphEdge],
]:
    edges = list(
        edges
    )

    if not edges:
        raise ValueError(
            "At least one road graph edge is required."
        )

    adjacency: dict[
        str,
        list[RoadGraphEdge],
    ] = {}

    for edge in edges:
        adjacency.setdefault(
            edge.start_node,
            [],
        ).append(
            edge
        )

        adjacency.setdefault(
            edge.end_node,
            [],
        )

    return adjacency


def _classify_route_safety(
    segments: tuple[
        RouteSegmentDecision,
        ...
    ],
) -> RouteSafetyLevel:
    if not segments:
        return RouteSafetyLevel.SAFE

    levels = {
        segment.risk_level
        for segment in segments
    }

    if (
        RoadFloodRiskLevel.HIGH
        in levels
        or RoadFloodRiskLevel.SEVERE
        in levels
    ):
        return RouteSafetyLevel.HIGH_RISK

    if (
        RoadFloodRiskLevel.WATCH
        in levels
        or RoadFloodRiskLevel.WARNING
        in levels
        or any(
            segment.recommendation
            == RoadRecommendation.CAUTION
            for segment in segments
        )
    ):
        return RouteSafetyLevel.CAUTION

    return RouteSafetyLevel.SAFE


def find_safest_route(
    *,
    start_node: str,
    destination_node: str,
    edges: Iterable[
        RoadGraphEdge
    ],
    policy: RoutingPolicy | None = None,
) -> SafeRouteResult:
    """
    Find the minimum generalized-risk route using Dijkstra's
    algorithm.

    Importantly, generalized cost is not equal to travel time.

    CLOSED roads never participate in route search.

    AVOID roads are excluded under the normal policy.

    The algorithm therefore naturally prefers routes that may
    take somewhat longer but have lower flood exposure and better
    confidence.
    """

    if policy is None:
        policy = RoutingPolicy()

    _validate_policy(
        policy
    )

    if not start_node.strip():
        raise ValueError(
            "start_node cannot be empty."
        )

    if not destination_node.strip():
        raise ValueError(
            "destination_node cannot be empty."
        )

    adjacency = _build_adjacency(
        edges
    )

    if start_node not in adjacency:
        raise RouteNotFoundError(
            f"Unknown start node: {start_node}"
        )

    if destination_node not in adjacency:
        raise RouteNotFoundError(
            "Unknown destination node: "
            f"{destination_node}"
        )

    if start_node == destination_node:
        return SafeRouteResult(
            start_node=start_node,
            destination_node=destination_node,
            node_path=(
                start_node,
            ),
            segments=(),
            travel_time_minutes=0.0,
            routing_cost=0.0,
            maximum_risk_score=0.0,
            minimum_confidence=1.0,
            safety_level=(
                RouteSafetyLevel.SAFE
            ),
            avoided_closed_segments=0,
            avoided_predicted_unsafe_segments=0,
            reasons=(
                "origin_is_destination",
            ),
        )

    distances: dict[
        str,
        float,
    ] = {
        node: float("inf")
        for node in adjacency
    }

    distances[
        start_node
    ] = 0.0

    previous: dict[
        str,
        tuple[
            str,
            RoadGraphEdge,
        ],
    ] = {}

    queue: list[
        tuple[
            float,
            str,
        ]
    ] = [
        (
            0.0,
            start_node,
        )
    ]

    while queue:
        (
            current_cost,
            current_node,
        ) = heapq.heappop(
            queue
        )

        if (
            current_cost
            > distances[
                current_node
            ]
        ):
            continue

        if (
            current_node
            == destination_node
        ):
            break

        for edge in adjacency[
            current_node
        ]:
            edge_cost = (
                calculate_edge_routing_cost(
                    edge,
                    policy,
                )
            )

            if not isfinite(
                edge_cost
            ):
                continue

            next_node = (
                edge.end_node
            )

            candidate_cost = (
                current_cost
                + edge_cost
            )

            if (
                candidate_cost
                < distances[
                    next_node
                ]
            ):
                distances[
                    next_node
                ] = candidate_cost

                previous[
                    next_node
                ] = (
                    current_node,
                    edge,
                )

                heapq.heappush(
                    queue,
                    (
                        candidate_cost,
                        next_node,
                    ),
                )

    if not isfinite(
        distances[
            destination_node
        ]
    ):
        raise RouteNotFoundError(
            "No routable path exists between "
            f"{start_node} and {destination_node}."
        )

    reversed_edges: list[
        RoadGraphEdge
    ] = []

    current = destination_node

    while current != start_node:
        if current not in previous:
            raise RouteNotFoundError(
                "Route reconstruction failed."
            )

        (
            previous_node,
            edge,
        ) = previous[
            current
        ]

        reversed_edges.append(
            edge
        )

        current = previous_node

    route_edges = list(
        reversed(
            reversed_edges
        )
    )

    node_path: list[str] = [
        start_node
    ]

    segment_decisions: list[
        RouteSegmentDecision
    ] = []

    for edge in route_edges:
        edge_cost = (
            calculate_edge_routing_cost(
                edge,
                policy,
            )
        )

        node_path.append(
            edge.end_node
        )

        segment_decisions.append(
            RouteSegmentDecision(
                road_id=edge.road_id,
                start_node=edge.start_node,
                end_node=edge.end_node,
                travel_time_minutes=(
                    edge.travel_time_minutes
                ),
                routing_cost=edge_cost,
                risk_score=(
                    edge.assessment.risk_score
                ),
                risk_level=(
                    edge.assessment.risk_level
                ),
                recommendation=(
                    edge.assessment.recommendation
                ),
                confidence=(
                    edge.assessment.data_confidence
                ),
                reasons=(
                    edge.assessment.reasons
                ),
            )
        )

    segments_tuple = tuple(
        segment_decisions
    )

    travel_time = sum(
        segment.travel_time_minutes
        for segment in segments_tuple
    )

    maximum_risk = max(
        (
            segment.risk_score
            for segment in segments_tuple
        ),
        default=0.0,
    )

    minimum_confidence = min(
        (
            segment.confidence
            for segment in segments_tuple
        ),
        default=1.0,
    )

    safety_level = (
        _classify_route_safety(
            segments_tuple
        )
    )

    all_edges = [
        edge
        for edge_list in adjacency.values()
        for edge in edge_list
    ]

    avoided_closed_segments = sum(
        edge.assessment.recommendation
        == RoadRecommendation.CLOSED
        for edge in all_edges
    )

    avoided_predicted_unsafe_segments = sum(
        edge.assessment.recommendation
        == RoadRecommendation.AVOID
        and not policy.allow_avoid_segments
        for edge in all_edges
    )

    reasons: list[str] = []

    if avoided_closed_segments > 0:
        reasons.append(
            "authority_closed_segments_excluded"
        )

    if (
        avoided_predicted_unsafe_segments
        > 0
    ):
        reasons.append(
            "predicted_unsafe_segments_excluded"
        )

    if minimum_confidence < 0.60:
        reasons.append(
            "route_contains_low_confidence_segment"
        )

    if (
        safety_level
        == RouteSafetyLevel.SAFE
    ):
        reasons.append(
            "selected_route_has_low_segment_risk"
        )

    elif (
        safety_level
        == RouteSafetyLevel.CAUTION
    ):
        reasons.append(
            "selected_route_requires_caution"
        )

    else:
        reasons.append(
            "selected_route_contains_high_risk_segment"
        )

    return SafeRouteResult(
        start_node=start_node,
        destination_node=destination_node,
        node_path=tuple(
            node_path
        ),
        segments=segments_tuple,
        travel_time_minutes=float(
            travel_time
        ),
        routing_cost=float(
            distances[
                destination_node
            ]
        ),
        maximum_risk_score=float(
            maximum_risk
        ),
        minimum_confidence=float(
            minimum_confidence
        ),
        safety_level=safety_level,
        avoided_closed_segments=int(
            avoided_closed_segments
        ),
        avoided_predicted_unsafe_segments=int(
            avoided_predicted_unsafe_segments
        ),
        reasons=tuple(
            reasons
        ),
    )