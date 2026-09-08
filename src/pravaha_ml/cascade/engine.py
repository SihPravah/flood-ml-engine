from collections import deque
from typing import Iterable

from pravaha_ml.cascade.models import (
    CascadeLink,
    CascadeRule,
    CascadeSeverity,
    HazardCascadeAssessment,
    HazardEvidenceStatus,
    HazardState,
    HazardType,
)


DEFAULT_CASCADE_RULES: tuple[
    CascadeRule,
    ...
] = (
    CascadeRule(
        rule_id="RAIN_TO_SOIL",
        source_hazard=HazardType.EXTREME_RAINFALL,
        target_hazard=HazardType.SOIL_SATURATION,
        minimum_source_intensity=0.50,
        transfer_factor=0.80,
        confidence_factor=0.90,
        reason_code="rainfall_can_increase_soil_saturation",
    ),
    CascadeRule(
        rule_id="RAIN_TO_FLASH_FLOOD",
        source_hazard=HazardType.EXTREME_RAINFALL,
        target_hazard=HazardType.FLASH_FLOOD,
        minimum_source_intensity=0.60,
        transfer_factor=0.85,
        confidence_factor=0.85,
        reason_code="extreme_rainfall_can_trigger_flash_flood",
    ),
    CascadeRule(
        rule_id="SOIL_TO_LANDSLIDE",
        source_hazard=HazardType.SOIL_SATURATION,
        target_hazard=HazardType.LANDSLIDE,
        minimum_source_intensity=0.70,
        transfer_factor=0.75,
        confidence_factor=0.80,
        reason_code="saturated_soil_can_increase_landslide_risk",
    ),
    CascadeRule(
        rule_id="FLASH_FLOOD_TO_DRAIN",
        source_hazard=HazardType.FLASH_FLOOD,
        target_hazard=HazardType.DRAIN_OVERLOAD,
        minimum_source_intensity=0.55,
        transfer_factor=0.85,
        confidence_factor=0.85,
        reason_code="flash_flood_runoff_can_overload_drainage",
    ),
    CascadeRule(
        rule_id="DRAIN_TO_ROAD",
        source_hazard=HazardType.DRAIN_OVERLOAD,
        target_hazard=HazardType.ROAD_FLOODING,
        minimum_source_intensity=0.60,
        transfer_factor=0.80,
        confidence_factor=0.85,
        reason_code="drain_overload_can_flood_nearby_roads",
    ),
    CascadeRule(
        rule_id="LANDSLIDE_TO_ROAD",
        source_hazard=HazardType.LANDSLIDE,
        target_hazard=HazardType.ROAD_OBSTRUCTION,
        minimum_source_intensity=0.70,
        transfer_factor=0.80,
        confidence_factor=0.75,
        reason_code="landslide_can_obstruct_road",
    ),
    CascadeRule(
        rule_id="LANDSLIDE_TO_STREAM_BLOCKAGE",
        source_hazard=HazardType.LANDSLIDE,
        target_hazard=HazardType.STREAM_BLOCKAGE,
        minimum_source_intensity=0.75,
        transfer_factor=0.70,
        confidence_factor=0.70,
        reason_code="landslide_can_block_stream_or_channel",
    ),
    CascadeRule(
        rule_id="STREAM_BLOCKAGE_TO_FLASH_FLOOD",
        source_hazard=HazardType.STREAM_BLOCKAGE,
        target_hazard=HazardType.FLASH_FLOOD,
        minimum_source_intensity=0.65,
        transfer_factor=0.75,
        confidence_factor=0.70,
        reason_code="stream_blockage_can_increase_local_flood_risk",
    ),
    CascadeRule(
        rule_id="ROAD_FLOOD_TO_ROUTE",
        source_hazard=HazardType.ROAD_FLOODING,
        target_hazard=HazardType.ROUTE_DEGRADATION,
        minimum_source_intensity=0.55,
        transfer_factor=0.85,
        confidence_factor=0.90,
        reason_code="road_flooding_can_degrade_route_viability",
    ),
    CascadeRule(
        rule_id="ROAD_OBSTRUCTION_TO_ROUTE",
        source_hazard=HazardType.ROAD_OBSTRUCTION,
        target_hazard=HazardType.ROUTE_DEGRADATION,
        minimum_source_intensity=0.55,
        transfer_factor=0.90,
        confidence_factor=0.90,
        reason_code="road_obstruction_can_degrade_route_viability",
    ),
)


def classify_cascade_severity(
    intensity_score: float,
) -> CascadeSeverity:
    if not 0.0 <= intensity_score <= 1.0:
        raise ValueError(
            "intensity_score must be between 0 and 1."
        )

    if intensity_score >= 0.85:
        return CascadeSeverity.SEVERE

    if intensity_score >= 0.70:
        return CascadeSeverity.HIGH

    if intensity_score >= 0.50:
        return CascadeSeverity.WARNING

    if intensity_score >= 0.30:
        return CascadeSeverity.WATCH

    return CascadeSeverity.LOW


def _validate_rules(
    rules: Iterable[
        CascadeRule
    ],
) -> tuple[
    CascadeRule,
    ...
]:
    rules = tuple(
        rules
    )

    rule_ids = [
        rule.rule_id
        for rule in rules
    ]

    if len(
        rule_ids
    ) != len(
        set(rule_ids)
    ):
        raise ValueError(
            "Cascade rule IDs must be unique."
        )

    return rules


def _hazard_key(
    state: HazardState,
) -> tuple[
    HazardType,
    str | None,
]:
    return (
        state.hazard_type,
        state.entity_id,
    )


def _merge_state(
    *,
    existing: HazardState | None,
    candidate: HazardState,
) -> HazardState:
    """
    Merge duplicate hazard states conservatively.

    Preserve the stronger intensity.

    Confidence is not allowed to increase merely because an
    inferred duplicate exists.
    """

    if existing is None:
        return candidate

    if (
        candidate.intensity_score
        > existing.intensity_score
    ):
        stronger = candidate
        weaker = existing
    else:
        stronger = existing
        weaker = candidate

    confidence = min(
        stronger.confidence,
        max(
            stronger.confidence,
            weaker.confidence,
        ),
    )

    reasons = tuple(
        dict.fromkeys(
            stronger.reasons
            + weaker.reasons
        )
    )

    return HazardState(
        hazard_type=(
            stronger.hazard_type
        ),
        intensity_score=(
            stronger.intensity_score
        ),
        confidence=float(
            confidence
        ),
        evidence_status=(
            stronger.evidence_status
        ),
        entity_id=(
            stronger.entity_id
        ),
        reasons=reasons,
    )


def _infer_target(
    *,
    source: HazardState,
    rule: CascadeRule,
) -> tuple[
    HazardState,
    CascadeLink,
]:
    target_intensity = min(
        source.intensity_score
        * rule.transfer_factor,
        1.0,
    )

    target_confidence = min(
        source.confidence
        * rule.confidence_factor,
        1.0,
    )

    target = HazardState(
        hazard_type=rule.target_hazard,
        intensity_score=float(
            target_intensity
        ),
        confidence=float(
            target_confidence
        ),
        evidence_status=(
            HazardEvidenceStatus.POSSIBLE_CASCADE
        ),
        entity_id=source.entity_id,
        reasons=(
            rule.reason_code,
        ),
    )

    link = CascadeLink(
        rule_id=rule.rule_id,
        source_hazard=(
            rule.source_hazard
        ),
        target_hazard=(
            rule.target_hazard
        ),
        source_entity_id=(
            source.entity_id
        ),
        target_intensity_score=float(
            target_intensity
        ),
        target_confidence=float(
            target_confidence
        ),
        status=(
            HazardEvidenceStatus.POSSIBLE_CASCADE
        ),
        reason=rule.reason_code,
    )

    return (
        target,
        link,
    )


def assess_hazard_cascade(
    *,
    hazards: Iterable[
        HazardState
    ],
    rules: Iterable[
        CascadeRule
    ] = DEFAULT_CASCADE_RULES,
    maximum_depth: int = 4,
) -> HazardCascadeAssessment:
    """
    Propagate plausible downstream consequences from current
    hazards.

    This engine deliberately produces POSSIBLE_CASCADE states.

    It does NOT convert inferred consequences into OBSERVED or
    AUTHORITY_CONFIRMED events.

    Example:

        predicted landslide susceptibility
            ↓
        possible road obstruction

    does NOT mean:

        road obstruction confirmed

    maximum_depth prevents indefinite feedback through cyclic
    hazard relationships.
    """

    if maximum_depth < 1:
        raise ValueError(
            "maximum_depth must be at least 1."
        )

    rules = _validate_rules(
        rules
    )

    input_hazards = tuple(
        hazards
    )

    state_map: dict[
        tuple[
            HazardType,
            str | None,
        ],
        HazardState,
    ] = {}

    for hazard in input_hazards:
        key = _hazard_key(
            hazard
        )

        state_map[key] = _merge_state(
            existing=state_map.get(
                key
            ),
            candidate=hazard,
        )

    rules_by_source: dict[
        HazardType,
        list[CascadeRule],
    ] = {}

    for rule in rules:
        rules_by_source.setdefault(
            rule.source_hazard,
            [],
        ).append(
            rule
        )

    queue = deque(
        (
            state,
            0,
        )
        for state in state_map.values()
    )

    inferred_keys: set[
        tuple[
            HazardType,
            str | None,
        ]
    ] = set()

    links: list[
        CascadeLink
    ] = []

    processed: set[
        tuple[
            HazardType,
            str | None,
            int,
        ]
    ] = set()

    while queue:
        (
            source,
            depth,
        ) = queue.popleft()

        if depth >= maximum_depth:
            continue

        process_key = (
            source.hazard_type,
            source.entity_id,
            depth,
        )

        if process_key in processed:
            continue

        processed.add(
            process_key
        )

        for rule in rules_by_source.get(
            source.hazard_type,
            [],
        ):
            if (
                source.intensity_score
                < rule.minimum_source_intensity
            ):
                continue

            (
                candidate,
                link,
            ) = _infer_target(
                source=source,
                rule=rule,
            )

            target_key = _hazard_key(
                candidate
            )

            existing = state_map.get(
                target_key
            )

            merged = _merge_state(
                existing=existing,
                candidate=candidate,
            )

            state_changed = (
                existing is None
                or merged.intensity_score
                > existing.intensity_score
                + 1e-12
            )

            state_map[
                target_key
            ] = merged

            links.append(
                link
            )

            if existing is None:
                inferred_keys.add(
                    target_key
                )

            if state_changed:
                queue.append(
                    (
                        merged,
                        depth + 1,
                    )
                )

    inferred_hazards = tuple(
        state_map[key]
        for key in inferred_keys
    )

    all_states = tuple(
        state_map.values()
    )

    maximum_intensity = max(
        (
            state.intensity_score
            for state in all_states
        ),
        default=0.0,
    )

    highest_severity = (
        classify_cascade_severity(
            maximum_intensity
        )
    )

    critical_reasons: list[
        str
    ] = []

    for state in inferred_hazards:
        if (
            state.intensity_score
            >= 0.70
        ):
            critical_reasons.extend(
                state.reasons
            )

    critical_reasons = list(
        dict.fromkeys(
            critical_reasons
        )
    )

    return HazardCascadeAssessment(
        input_hazards=input_hazards,
        inferred_hazards=(
            inferred_hazards
        ),
        links=tuple(
            links
        ),
        highest_severity=(
            highest_severity
        ),
        critical_reasons=tuple(
            critical_reasons
        ),
    )