import pytest

from pravaha_ml.cascade.engine import (
    assess_hazard_cascade,
    classify_cascade_severity,
)
from pravaha_ml.cascade.models import (
    CascadeRule,
    CascadeSeverity,
    HazardEvidenceStatus,
    HazardState,
    HazardType,
)


def make_hazard(
    hazard_type: HazardType,
    *,
    intensity: float,
    confidence: float = 0.90,
    entity_id: str = "ZONE_001",
) -> HazardState:
    return HazardState(
        hazard_type=hazard_type,
        intensity_score=intensity,
        confidence=confidence,
        evidence_status=(
            HazardEvidenceStatus.PREDICTED
        ),
        entity_id=entity_id,
    )


def find_inferred(
    result,
    hazard_type: HazardType,
):
    return next(
        state
        for state in result.inferred_hazards
        if state.hazard_type
        == hazard_type
    )


def test_extreme_rainfall_can_trigger_flash_flood_cascade():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.EXTREME_RAINFALL,
                intensity=0.90,
            )
        ]
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.FLASH_FLOOD
        in inferred_types
    )


def test_flash_flood_can_propagate_to_drain_overload():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.FLASH_FLOOD,
                intensity=0.90,
            )
        ]
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.DRAIN_OVERLOAD
        in inferred_types
    )


def test_drain_overload_can_propagate_to_road_flooding():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.DRAIN_OVERLOAD,
                intensity=0.90,
            )
        ]
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.ROAD_FLOODING
        in inferred_types
    )


def test_landslide_can_propagate_to_road_obstruction():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.LANDSLIDE,
                intensity=0.95,
            )
        ]
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.ROAD_OBSTRUCTION
        in inferred_types
    )


def test_landslide_can_propagate_to_stream_blockage():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.LANDSLIDE,
                intensity=0.95,
            )
        ]
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.STREAM_BLOCKAGE
        in inferred_types
    )


def test_possible_cascade_does_not_become_observed():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.LANDSLIDE,
                intensity=0.95,
            )
        ]
    )

    obstruction = find_inferred(
        result,
        HazardType.ROAD_OBSTRUCTION,
    )

    assert (
        obstruction.evidence_status
        == HazardEvidenceStatus.POSSIBLE_CASCADE
    )


def test_inferred_confidence_decreases_downstream():
    source = make_hazard(
        HazardType.FLASH_FLOOD,
        intensity=0.90,
        confidence=0.90,
    )

    result = assess_hazard_cascade(
        hazards=[
            source
        ]
    )

    drain = find_inferred(
        result,
        HazardType.DRAIN_OVERLOAD,
    )

    assert (
        drain.confidence
        < source.confidence
    )


def test_below_threshold_source_does_not_trigger_rule():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.DRAIN_OVERLOAD,
                intensity=0.20,
            )
        ]
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.ROAD_FLOODING
        not in inferred_types
    )


def test_multistep_cascade_can_reach_route_degradation():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.FLASH_FLOOD,
                intensity=1.0,
            )
        ],
        maximum_depth=4,
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.ROUTE_DEGRADATION
        in inferred_types
    )


def test_entity_id_propagates_through_cascade():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.LANDSLIDE,
                intensity=0.95,
                entity_id="LS_018",
            )
        ]
    )

    obstruction = find_inferred(
        result,
        HazardType.ROAD_OBSTRUCTION,
    )

    assert (
        obstruction.entity_id
        == "LS_018"
    )


def test_maximum_depth_limits_propagation():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.FLASH_FLOOD,
                intensity=1.0,
            )
        ],
        maximum_depth=1,
    )

    inferred_types = {
        state.hazard_type
        for state in result.inferred_hazards
    }

    assert (
        HazardType.DRAIN_OVERLOAD
        in inferred_types
    )

    assert (
        HazardType.ROAD_FLOODING
        not in inferred_types
    )


def test_duplicate_rule_ids_rejected():
    rule = CascadeRule(
        rule_id="DUPLICATE",
        source_hazard=(
            HazardType.FLASH_FLOOD
        ),
        target_hazard=(
            HazardType.DRAIN_OVERLOAD
        ),
        minimum_source_intensity=0.50,
        transfer_factor=0.80,
        confidence_factor=0.80,
        reason_code="test",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Cascade rule IDs must be unique"
        ),
    ):
        assess_hazard_cascade(
            hazards=[
                make_hazard(
                    HazardType.FLASH_FLOOD,
                    intensity=0.90,
                )
            ],
            rules=(
                rule,
                rule,
            ),
        )


def test_invalid_maximum_depth_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "maximum_depth must be at least 1"
        ),
    ):
        assess_hazard_cascade(
            hazards=[],
            maximum_depth=0,
        )


def test_high_intensity_produces_high_or_severe_city_cascade():
    result = assess_hazard_cascade(
        hazards=[
            make_hazard(
                HazardType.FLASH_FLOOD,
                intensity=0.90,
            )
        ]
    )

    assert (
        result.highest_severity
        in {
            CascadeSeverity.HIGH,
            CascadeSeverity.SEVERE,
        }
    )


def test_severity_boundaries():
    assert (
        classify_cascade_severity(0.10)
        == CascadeSeverity.LOW
    )

    assert (
        classify_cascade_severity(0.35)
        == CascadeSeverity.WATCH
    )

    assert (
        classify_cascade_severity(0.55)
        == CascadeSeverity.WARNING
    )

    assert (
        classify_cascade_severity(0.75)
        == CascadeSeverity.HIGH
    )

    assert (
        classify_cascade_severity(0.90)
        == CascadeSeverity.SEVERE
    )