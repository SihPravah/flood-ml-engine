import pytest

from pravaha_ml.roads.models import (
    RoadFloodAssessment,
    RoadFloodRiskLevel,
    RoadRecommendation,
)
from pravaha_ml.roads.routing import (
    RoadGraphEdge,
    RouteNotFoundError,
    RouteSafetyLevel,
    RoutingPolicy,
    calculate_edge_routing_cost,
    edge_is_routable,
    find_safest_route,
)


def make_assessment(
    road_id: str,
    *,
    risk_score: float = 0.10,
    risk_level: RoadFloodRiskLevel = (
        RoadFloodRiskLevel.LOW
    ),
    recommendation: RoadRecommendation = (
        RoadRecommendation.PASSABLE
    ),
    confidence: float = 0.95,
) -> RoadFloodAssessment:
    return RoadFloodAssessment(
        road_id=road_id,
        risk_score=risk_score,
        risk_level=risk_level,
        recommendation=recommendation,
        data_confidence=confidence,
        drainage_exposure_score=risk_score,
        catchment_component=risk_score,
        terrain_component=risk_score,
        historical_component=risk_score,
        stream_component=risk_score,
        road_vulnerability_component=risk_score,
        nearest_drain_distance_m=25.0,
        inferred_avoidance=(
            recommendation
            == RoadRecommendation.AVOID
        ),
        authoritative_closure=(
            recommendation
            == RoadRecommendation.CLOSED
        ),
        reasons=(),
    )


def make_edge(
    start: str,
    end: str,
    road_id: str,
    travel_time: float,
    *,
    risk_score: float = 0.10,
    risk_level: RoadFloodRiskLevel = (
        RoadFloodRiskLevel.LOW
    ),
    recommendation: RoadRecommendation = (
        RoadRecommendation.PASSABLE
    ),
    confidence: float = 0.95,
) -> RoadGraphEdge:
    return RoadGraphEdge(
        start_node=start,
        end_node=end,
        road_id=road_id,
        travel_time_minutes=travel_time,
        assessment=make_assessment(
            road_id,
            risk_score=risk_score,
            risk_level=risk_level,
            recommendation=recommendation,
            confidence=confidence,
        ),
    )


def test_low_risk_edge_has_finite_cost():
    edge = make_edge(
        "A",
        "B",
        "R_1",
        5.0,
    )

    cost = calculate_edge_routing_cost(
        edge
    )

    assert cost > 5.0
    assert cost < float("inf")


def test_closed_edge_is_not_routable():
    edge = make_edge(
        "A",
        "B",
        "R_1",
        5.0,
        recommendation=(
            RoadRecommendation.CLOSED
        ),
    )

    policy = RoutingPolicy()

    assert (
        edge_is_routable(
            edge,
            policy,
        )
        is False
    )

    assert (
        calculate_edge_routing_cost(
            edge,
            policy,
        )
        == float("inf")
    )


def test_avoid_edge_is_excluded_by_default():
    edge = make_edge(
        "A",
        "B",
        "R_1",
        5.0,
        risk_score=0.80,
        risk_level=(
            RoadFloodRiskLevel.HIGH
        ),
        recommendation=(
            RoadRecommendation.AVOID
        ),
    )

    assert (
        edge_is_routable(
            edge,
            RoutingPolicy(),
        )
        is False
    )


def test_avoid_edge_can_be_allowed_for_fallback():
    edge = make_edge(
        "A",
        "B",
        "R_1",
        5.0,
        risk_score=0.80,
        risk_level=(
            RoadFloodRiskLevel.HIGH
        ),
        recommendation=(
            RoadRecommendation.AVOID
        ),
    )

    policy = RoutingPolicy(
        allow_avoid_segments=True
    )

    assert (
        edge_is_routable(
            edge,
            policy,
        )
        is True
    )

    assert (
        calculate_edge_routing_cost(
            edge,
            policy,
        )
        > edge.travel_time_minutes
    )


def test_safer_longer_route_is_preferred():
    edges = [
        make_edge(
            "A",
            "B",
            "R_FAST_1",
            3.0,
            risk_score=0.65,
            risk_level=(
                RoadFloodRiskLevel.WARNING
            ),
            recommendation=(
                RoadRecommendation.CAUTION
            ),
        ),
        make_edge(
            "B",
            "D",
            "R_FAST_2",
            3.0,
            risk_score=0.65,
            risk_level=(
                RoadFloodRiskLevel.WARNING
            ),
            recommendation=(
                RoadRecommendation.CAUTION
            ),
        ),
        make_edge(
            "A",
            "C",
            "R_SAFE_1",
            5.0,
            risk_score=0.10,
            risk_level=(
                RoadFloodRiskLevel.LOW
            ),
        ),
        make_edge(
            "C",
            "D",
            "R_SAFE_2",
            5.0,
            risk_score=0.10,
            risk_level=(
                RoadFloodRiskLevel.LOW
            ),
        ),
    ]

    result = find_safest_route(
        start_node="A",
        destination_node="D",
        edges=edges,
    )

    assert (
        result.node_path
        == (
            "A",
            "C",
            "D",
        )
    )

    assert (
        result.travel_time_minutes
        == pytest.approx(10.0)
    )

    assert (
        result.safety_level
        == RouteSafetyLevel.SAFE
    )


def test_closed_shortcut_is_excluded():
    edges = [
        make_edge(
            "A",
            "D",
            "R_CLOSED",
            2.0,
            recommendation=(
                RoadRecommendation.CLOSED
            ),
        ),
        make_edge(
            "A",
            "B",
            "R_1",
            5.0,
        ),
        make_edge(
            "B",
            "D",
            "R_2",
            5.0,
        ),
    ]

    result = find_safest_route(
        start_node="A",
        destination_node="D",
        edges=edges,
    )

    assert (
        result.node_path
        == (
            "A",
            "B",
            "D",
        )
    )

    assert (
        result.avoided_closed_segments
        == 1
    )

    assert (
        "authority_closed_segments_excluded"
        in result.reasons
    )


def test_predicted_avoid_segment_is_excluded():
    edges = [
        make_edge(
            "A",
            "D",
            "R_UNSAFE",
            2.0,
            risk_score=0.90,
            risk_level=(
                RoadFloodRiskLevel.SEVERE
            ),
            recommendation=(
                RoadRecommendation.AVOID
            ),
        ),
        make_edge(
            "A",
            "B",
            "R_SAFE_1",
            5.0,
        ),
        make_edge(
            "B",
            "D",
            "R_SAFE_2",
            5.0,
        ),
    ]

    result = find_safest_route(
        start_node="A",
        destination_node="D",
        edges=edges,
    )

    assert (
        result.node_path
        == (
            "A",
            "B",
            "D",
        )
    )

    assert (
        result.avoided_predicted_unsafe_segments
        == 1
    )


def test_low_confidence_route_gets_penalized():
    high_confidence = make_edge(
        "A",
        "B",
        "R_HIGH_CONF",
        5.0,
        confidence=0.95,
    )

    low_confidence = make_edge(
        "A",
        "B",
        "R_LOW_CONF",
        5.0,
        confidence=0.20,
    )

    policy = RoutingPolicy()

    assert (
        calculate_edge_routing_cost(
            low_confidence,
            policy,
        )
        >
        calculate_edge_routing_cost(
            high_confidence,
            policy,
        )
    )


def test_higher_risk_edge_gets_higher_cost():
    low = make_edge(
        "A",
        "B",
        "R_LOW",
        5.0,
        risk_score=0.10,
        risk_level=(
            RoadFloodRiskLevel.LOW
        ),
    )

    warning = make_edge(
        "A",
        "B",
        "R_WARNING",
        5.0,
        risk_score=0.60,
        risk_level=(
            RoadFloodRiskLevel.WARNING
        ),
        recommendation=(
            RoadRecommendation.CAUTION
        ),
    )

    assert (
        calculate_edge_routing_cost(
            warning
        )
        >
        calculate_edge_routing_cost(
            low
        )
    )


def test_route_reports_maximum_risk():
    edges = [
        make_edge(
            "A",
            "B",
            "R_1",
            5.0,
            risk_score=0.10,
        ),
        make_edge(
            "B",
            "D",
            "R_2",
            5.0,
            risk_score=0.40,
            risk_level=(
                RoadFloodRiskLevel.WATCH
            ),
            recommendation=(
                RoadRecommendation.CAUTION
            ),
        ),
    ]

    result = find_safest_route(
        start_node="A",
        destination_node="D",
        edges=edges,
    )

    assert (
        result.maximum_risk_score
        == pytest.approx(0.40)
    )

    assert (
        result.safety_level
        == RouteSafetyLevel.CAUTION
    )


def test_route_reports_minimum_confidence():
    edges = [
        make_edge(
            "A",
            "B",
            "R_1",
            5.0,
            confidence=0.90,
        ),
        make_edge(
            "B",
            "D",
            "R_2",
            5.0,
            confidence=0.55,
            recommendation=(
                RoadRecommendation.CAUTION
            ),
        ),
    ]

    result = find_safest_route(
        start_node="A",
        destination_node="D",
        edges=edges,
    )

    assert (
        result.minimum_confidence
        == pytest.approx(0.55)
    )

    assert (
        "route_contains_low_confidence_segment"
        in result.reasons
    )


def test_no_route_when_all_paths_blocked():
    edges = [
        make_edge(
            "A",
            "B",
            "R_1",
            5.0,
            recommendation=(
                RoadRecommendation.CLOSED
            ),
        ),
        make_edge(
            "B",
            "D",
            "R_2",
            5.0,
        ),
    ]

    with pytest.raises(
        RouteNotFoundError,
        match="No routable path exists",
    ):
        find_safest_route(
            start_node="A",
            destination_node="D",
            edges=edges,
        )


def test_origin_equals_destination():
    edges = [
        make_edge(
            "A",
            "B",
            "R_1",
            5.0,
        )
    ]

    result = find_safest_route(
        start_node="A",
        destination_node="A",
        edges=edges,
    )

    assert (
        result.node_path
        == ("A",)
    )

    assert (
        result.travel_time_minutes
        == pytest.approx(0.0)
    )

    assert (
        result.safety_level
        == RouteSafetyLevel.SAFE
    )


def test_unknown_destination_rejected():
    edges = [
        make_edge(
            "A",
            "B",
            "R_1",
            5.0,
        )
    ]

    with pytest.raises(
        RouteNotFoundError,
        match="Unknown destination node",
    ):
        find_safest_route(
            start_node="A",
            destination_node="UNKNOWN",
            edges=edges,
        )


def test_invalid_edge_travel_time_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "travel_time_minutes must be greater than 0"
        ),
    ):
        make_edge(
            "A",
            "B",
            "R_1",
            0.0,
        )


def test_edge_and_assessment_road_id_must_match():
    assessment = make_assessment(
        "R_OTHER"
    )

    with pytest.raises(
        ValueError,
        match=(
            "road_id must match assessment.road_id"
        ),
    ):
        RoadGraphEdge(
            start_node="A",
            end_node="B",
            road_id="R_1",
            travel_time_minutes=5.0,
            assessment=assessment,
        )


def test_negative_policy_penalty_rejected():
    policy = RoutingPolicy(
        risk_penalty_minutes=-1.0
    )

    edge = make_edge(
        "A",
        "B",
        "R_1",
        5.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing penalties cannot be negative"
        ),
    ):
        calculate_edge_routing_cost(
            edge,
            policy,
        )