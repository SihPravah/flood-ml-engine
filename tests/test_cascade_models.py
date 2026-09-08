import pytest

from pravaha_ml.cascade.models import (
    CascadeRule,
    HazardEvidenceStatus,
    HazardState,
    HazardType,
)


def test_valid_hazard_state():
    state = HazardState(
        hazard_type=(
            HazardType.FLASH_FLOOD
        ),
        intensity_score=0.70,
        confidence=0.80,
        evidence_status=(
            HazardEvidenceStatus.PREDICTED
        ),
        entity_id="C_001",
    )

    assert (
        state.hazard_type
        == HazardType.FLASH_FLOOD
    )


def test_invalid_hazard_intensity_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "intensity_score must be between 0 and 1"
        ),
    ):
        HazardState(
            hazard_type=(
                HazardType.FLASH_FLOOD
            ),
            intensity_score=1.20,
            confidence=0.80,
            evidence_status=(
                HazardEvidenceStatus.PREDICTED
            ),
        )


def test_invalid_hazard_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "confidence must be between 0 and 1"
        ),
    ):
        HazardState(
            hazard_type=(
                HazardType.FLASH_FLOOD
            ),
            intensity_score=0.70,
            confidence=-0.10,
            evidence_status=(
                HazardEvidenceStatus.PREDICTED
            ),
        )


def test_empty_entity_id_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "entity_id cannot be empty"
        ),
    ):
        HazardState(
            hazard_type=(
                HazardType.FLASH_FLOOD
            ),
            intensity_score=0.70,
            confidence=0.80,
            evidence_status=(
                HazardEvidenceStatus.PREDICTED
            ),
            entity_id="",
        )


def test_valid_cascade_rule():
    rule = CascadeRule(
        rule_id="TEST",
        source_hazard=(
            HazardType.FLASH_FLOOD
        ),
        target_hazard=(
            HazardType.DRAIN_OVERLOAD
        ),
        minimum_source_intensity=0.50,
        transfer_factor=0.80,
        confidence_factor=0.90,
        reason_code="test_reason",
    )

    assert rule.rule_id == "TEST"


def test_invalid_transfer_factor_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "transfer_factor must be between 0 and 1"
        ),
    ):
        CascadeRule(
            rule_id="TEST",
            source_hazard=(
                HazardType.FLASH_FLOOD
            ),
            target_hazard=(
                HazardType.DRAIN_OVERLOAD
            ),
            minimum_source_intensity=0.50,
            transfer_factor=1.50,
            confidence_factor=0.90,
            reason_code="test_reason",
        )