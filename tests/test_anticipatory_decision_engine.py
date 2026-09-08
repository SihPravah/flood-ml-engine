import pytest

from pravaha_ml.decision.engine import (
    assess_anticipatory_decision,
)
from pravaha_ml.decision.models import (
    AnticipatoryDecisionInputs,
    AnticipatoryDecisionPolicy,
    AuthorityEventStatus,
    DecisionConfidenceLevel,
    OperationalState,
)


def make_inputs(
    **overrides,
) -> AnticipatoryDecisionInputs:
    values = {
        "unit_id": "WARD_03",
        "impact_score": 0.15,
        "trajectory_rising": False,
        "trajectory_rapidly_rising": False,
        "projected_high_threshold_minutes": None,
        "forecast_worst_consequence_score": 0.20,
        "forecast_worsening": False,
        "cascade_intensity_score": 0.15,
        "evacuation_ready": True,
        "evacuation_clearance_minutes": 20.0,
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


def test_quiet_conditions_return_normal():
    result = assess_anticipatory_decision(
        inputs=make_inputs()
    )

    assert (
        result.operational_state
        == OperationalState.NORMAL
    )


def test_elevated_impact_returns_watch():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            impact_score=0.35
        )
    )

    assert (
        result.operational_state
        == OperationalState.WATCH
    )


def test_rising_trajectory_returns_watch():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            trajectory_rising=True
        )
    )

    assert (
        result.operational_state
        == OperationalState.WATCH
    )


def test_rapidly_rising_trajectory_can_trigger_prepare():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            trajectory_rising=True,
            trajectory_rapidly_rising=True,
        )
    )

    assert (
        result.operational_state
        == OperationalState.PREPARE
    )


def test_projected_high_within_prepare_window_triggers_prepare():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            projected_high_threshold_minutes=55.0
        )
    )

    assert (
        result.operational_state
        == OperationalState.PREPARE
    )


def test_warning_impact_returns_warning():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            impact_score=0.60
        )
    )

    assert (
        result.operational_state
        == OperationalState.WARNING
    )


def test_worsening_forecast_with_high_consequence_returns_warning():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            forecast_worsening=True,
            forecast_worst_consequence_score=0.75,
        )
    )

    assert (
        result.operational_state
        == OperationalState.WARNING
    )


def test_large_cascade_returns_warning():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            cascade_intensity_score=0.75
        )
    )

    assert (
        result.operational_state
        == OperationalState.WARNING
    )


def test_negative_clearance_margin_can_trigger_evacuation_support():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            impact_score=0.75,
            evacuation_clearance_minutes=50.0,
            hazard_window_minutes=40.0,
        )
    )

    assert (
        result.operational_state
        == OperationalState.EVACUATION_SUPPORT
    )

    assert (
        result.evacuation_margin_minutes
        == pytest.approx(-10.0)
    )


def test_near_threshold_and_not_ready_triggers_evacuation_support():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            impact_score=0.45,
            projected_high_threshold_minutes=30.0,
            evacuation_ready=False,
        )
    )

    assert (
        result.operational_state
        == OperationalState.EVACUATION_SUPPORT
    )


def test_severe_forecast_and_tight_margin_triggers_evacuation_support():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            forecast_worsening=True,
            forecast_worst_consequence_score=0.90,
            evacuation_clearance_minutes=45.0,
            hazard_window_minutes=50.0,
        )
    )

    assert (
        result.operational_state
        == OperationalState.EVACUATION_SUPPORT
    )


def test_observed_event_returns_active_event():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            event_observed=True
        )
    )

    assert (
        result.operational_state
        == OperationalState.ACTIVE_EVENT
    )


def test_authority_confirmed_event_returns_active_event():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            authority_event_status=(
                AuthorityEventStatus.EVENT_CONFIRMED
            )
        )
    )

    assert (
        result.operational_state
        == OperationalState.ACTIVE_EVENT
    )


def test_authority_evacuation_order_returns_active_event():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            authority_event_status=(
                AuthorityEventStatus.EVACUATION_ORDER
            )
        )
    )

    assert (
        result.operational_state
        == OperationalState.ACTIVE_EVENT
    )


def test_prediction_alone_does_not_become_active_event():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            impact_score=0.95,
            forecast_worsening=True,
            forecast_worst_consequence_score=0.95,
            cascade_intensity_score=0.90,
            event_observed=False,
            authority_event_status=(
                AuthorityEventStatus.NONE
            ),
        )
    )

    assert (
        result.operational_state
        != OperationalState.ACTIVE_EVENT
    )


def test_low_confidence_returns_insufficient_data():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            system_confidence=0.40
        )
    )

    assert (
        result.operational_state
        == OperationalState.INSUFFICIENT_DATA
    )

    assert (
        "decision_confidence_insufficient"
        in result.reasons
    )


def test_low_confidence_does_not_claim_normal():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            impact_score=0.05,
            system_confidence=0.30,
        )
    )

    assert (
        result.operational_state
        != OperationalState.NORMAL
    )


def test_confidence_level_high():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            system_confidence=0.90
        )
    )

    assert (
        result.confidence_level
        == DecisionConfidenceLevel.HIGH
    )


def test_confidence_level_moderate():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            system_confidence=0.70
        )
    )

    assert (
        result.confidence_level
        == DecisionConfidenceLevel.MODERATE
    )


def test_warning_authority_status_can_trigger_warning():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            authority_event_status=(
                AuthorityEventStatus.WARNING
            )
        )
    )

    assert (
        result.operational_state
        == OperationalState.WARNING
    )


def test_authority_watch_can_trigger_prepare():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            authority_event_status=(
                AuthorityEventStatus.WATCH
            )
        )
    )

    assert (
        result.operational_state
        == OperationalState.PREPARE
    )


def test_clearance_margin_is_exposed():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            evacuation_clearance_minutes=35.0,
            hazard_window_minutes=60.0,
        )
    )

    assert (
        result.evacuation_margin_minutes
        == pytest.approx(25.0)
    )


def test_unknown_clearance_margin_is_preserved():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            evacuation_clearance_minutes=None,
            hazard_window_minutes=None,
        )
    )

    assert (
        result.evacuation_margin_minutes
        is None
    )

    assert (
        "evacuation_clearance_margin_unknown"
        in result.reasons
    )


def test_evacuation_support_is_not_authority_order():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            projected_high_threshold_minutes=20.0,
            evacuation_ready=False,
        )
    )

    assert (
        result.operational_state
        == OperationalState.EVACUATION_SUPPORT
    )

    assert (
        result.authority_event_status
        == AuthorityEventStatus.NONE
    )


def test_recommendation_codes_are_exposed():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            projected_high_threshold_minutes=50.0
        )
    )

    assert (
        result.recommendation_code
        == "BEGIN_PREPARATORY_ACTIONS"
    )


def test_invalid_policy_impact_threshold_order_rejected():
    policy = AnticipatoryDecisionPolicy(
        watch_impact_threshold=0.60,
        warning_impact_threshold=0.40,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Impact thresholds must be ordered"
        ),
    ):
        assess_anticipatory_decision(
            inputs=make_inputs(),
            policy=policy,
        )


def test_invalid_horizon_order_rejected():
    policy = AnticipatoryDecisionPolicy(
        prepare_horizon_minutes=30.0,
        evacuation_support_horizon_minutes=60.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "evacuation_support_horizon_minutes cannot exceed"
        ),
    ):
        assess_anticipatory_decision(
            inputs=make_inputs(),
            policy=policy,
        )


def test_evacuation_support_reason_is_exposed():
    result = assess_anticipatory_decision(
        inputs=make_inputs(
            projected_high_threshold_minutes=20.0,
            evacuation_ready=False,
        )
    )

    assert (
        "anticipatory_evacuation_support_conditions_met"
        in result.reasons
    )