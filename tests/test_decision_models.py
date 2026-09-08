import pytest

from pravaha_ml.decision.models import (
    AnticipatoryDecisionInputs,
    AuthorityEventStatus,
)


def make_inputs(
    **overrides,
) -> AnticipatoryDecisionInputs:
    values = {
        "unit_id": "WARD_03",
        "impact_score": 0.30,
        "trajectory_rising": False,
        "trajectory_rapidly_rising": False,
        "projected_high_threshold_minutes": None,
        "forecast_worst_consequence_score": 0.30,
        "forecast_worsening": False,
        "cascade_intensity_score": 0.20,
        "evacuation_ready": True,
        "evacuation_clearance_minutes": 30.0,
        "hazard_window_minutes": 90.0,
        "authority_event_status": (
            AuthorityEventStatus.NONE
        ),
        "event_observed": False,
        "system_confidence": 0.90,
    }

    values.update(
        overrides
    )

    return AnticipatoryDecisionInputs(
        **values
    )


def test_valid_decision_inputs():
    inputs = make_inputs()

    assert (
        inputs.unit_id
        == "WARD_03"
    )


def test_empty_unit_id_rejected():
    with pytest.raises(
        ValueError,
        match="unit_id cannot be empty",
    ):
        make_inputs(
            unit_id=""
        )


def test_invalid_impact_score_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "impact_score must be between 0 and 1"
        ),
    ):
        make_inputs(
            impact_score=1.20
        )


def test_invalid_forecast_score_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "forecast_worst_consequence_score "
            "must be between 0 and 1"
        ),
    ):
        make_inputs(
            forecast_worst_consequence_score=-0.1
        )


def test_negative_projected_threshold_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "projected_high_threshold_minutes "
            "cannot be negative"
        ),
    ):
        make_inputs(
            projected_high_threshold_minutes=-1.0
        )


def test_negative_clearance_time_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "evacuation_clearance_minutes "
            "cannot be negative"
        ),
    ):
        make_inputs(
            evacuation_clearance_minutes=-5.0
        )


def test_invalid_system_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "system_confidence must be between 0 and 1"
        ),
    ):
        make_inputs(
            system_confidence=1.50
        )