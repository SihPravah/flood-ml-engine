from pravaha_ml.decision.models import (
    AnticipatoryDecision,
    AnticipatoryDecisionInputs,
    AnticipatoryDecisionPolicy,
    AuthorityEventStatus,
    DecisionConfidenceLevel,
    OperationalState,
)


def _validate_policy(
    policy: AnticipatoryDecisionPolicy,
) -> None:
    normalized = {
        "minimum_reliable_confidence": (
            policy.minimum_reliable_confidence
        ),
        "watch_impact_threshold": (
            policy.watch_impact_threshold
        ),
        "warning_impact_threshold": (
            policy.warning_impact_threshold
        ),
        "high_impact_threshold": (
            policy.high_impact_threshold
        ),
        "severe_forecast_threshold": (
            policy.severe_forecast_threshold
        ),
        "significant_cascade_threshold": (
            policy.significant_cascade_threshold
        ),
        "high_confidence_threshold": (
            policy.high_confidence_threshold
        ),
        "moderate_confidence_threshold": (
            policy.moderate_confidence_threshold
        ),
    }

    for field_name, value in normalized.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

    if not (
        policy.watch_impact_threshold
        <= policy.warning_impact_threshold
        <= policy.high_impact_threshold
    ):
        raise ValueError(
            "Impact thresholds must be ordered."
        )

    if (
        policy.prepare_horizon_minutes
        <= 0.0
    ):
        raise ValueError(
            "prepare_horizon_minutes must be greater than 0."
        )

    if (
        policy.evacuation_support_horizon_minutes
        <= 0.0
    ):
        raise ValueError(
            "evacuation_support_horizon_minutes must be "
            "greater than 0."
        )

    if (
        policy.evacuation_support_horizon_minutes
        > policy.prepare_horizon_minutes
    ):
        raise ValueError(
            "evacuation_support_horizon_minutes cannot exceed "
            "prepare_horizon_minutes."
        )

    if (
        policy.tight_clearance_margin_minutes
        < 0.0
    ):
        raise ValueError(
            "tight_clearance_margin_minutes cannot be negative."
        )

    if not (
        policy.minimum_reliable_confidence
        <= policy.moderate_confidence_threshold
        <= policy.high_confidence_threshold
    ):
        raise ValueError(
            "Decision confidence thresholds must be ordered."
        )


def _confidence_level(
    confidence: float,
    policy: AnticipatoryDecisionPolicy,
) -> DecisionConfidenceLevel:
    if (
        confidence
        >= policy.high_confidence_threshold
    ):
        return DecisionConfidenceLevel.HIGH

    if (
        confidence
        >= policy.moderate_confidence_threshold
    ):
        return DecisionConfidenceLevel.MODERATE

    if confidence > 0.0:
        return DecisionConfidenceLevel.LOW

    return DecisionConfidenceLevel.INSUFFICIENT


def _clearance_margin(
    inputs: AnticipatoryDecisionInputs,
) -> float | None:
    if (
        inputs.evacuation_clearance_minutes is None
        or inputs.hazard_window_minutes is None
    ):
        return None

    return float(
        inputs.hazard_window_minutes
        - inputs.evacuation_clearance_minutes
    )


def _active_event_present(
    inputs: AnticipatoryDecisionInputs,
) -> bool:
    return (
        inputs.event_observed
        or inputs.authority_event_status
        in {
            AuthorityEventStatus.EVENT_CONFIRMED,
            AuthorityEventStatus.EVACUATION_ORDER,
        }
    )


def _evacuation_support_needed(
    *,
    inputs: AnticipatoryDecisionInputs,
    margin: float | None,
    policy: AnticipatoryDecisionPolicy,
) -> bool:
    threshold_soon = (
        inputs.projected_high_threshold_minutes
        is not None
        and inputs.projected_high_threshold_minutes
        <= policy.evacuation_support_horizon_minutes
    )

    clearance_pressure = (
        margin is not None
        and margin
        <= policy.tight_clearance_margin_minutes
    )

    high_current_impact = (
        inputs.impact_score
        >= policy.high_impact_threshold
    )

    severe_forecast = (
        inputs.forecast_worsening
        and inputs.forecast_worst_consequence_score
        >= policy.severe_forecast_threshold
    )

    significant_cascade = (
        inputs.cascade_intensity_score
        >= policy.significant_cascade_threshold
    )

    return (
        (
            threshold_soon
            and (
                clearance_pressure
                or not inputs.evacuation_ready
            )
        )
        or (
            high_current_impact
            and clearance_pressure
        )
        or (
            severe_forecast
            and (
                threshold_soon
                or clearance_pressure
            )
        )
        or (
            significant_cascade
            and clearance_pressure
        )
    )


def _warning_needed(
    *,
    inputs: AnticipatoryDecisionInputs,
    policy: AnticipatoryDecisionPolicy,
) -> bool:
    return (
        inputs.impact_score
        >= policy.warning_impact_threshold
        or (
            inputs.trajectory_rapidly_rising
            and inputs.impact_score
            >= policy.watch_impact_threshold
        )
        or (
            inputs.forecast_worsening
            and inputs.forecast_worst_consequence_score
            >= policy.high_impact_threshold
        )
        or (
            inputs.cascade_intensity_score
            >= policy.high_impact_threshold
        )
        or (
            inputs.authority_event_status
            == AuthorityEventStatus.WARNING
        )
    )


def _prepare_needed(
    *,
    inputs: AnticipatoryDecisionInputs,
    policy: AnticipatoryDecisionPolicy,
) -> bool:
    threshold_approaching = (
        inputs.projected_high_threshold_minutes
        is not None
        and inputs.projected_high_threshold_minutes
        <= policy.prepare_horizon_minutes
    )

    return (
        threshold_approaching
        or inputs.trajectory_rapidly_rising
        or (
            inputs.trajectory_rising
            and inputs.forecast_worsening
        )
        or (
            inputs.impact_score
            >= policy.warning_impact_threshold
        )
        or (
            inputs.authority_event_status
            == AuthorityEventStatus.WATCH
        )
    )


def _watch_needed(
    *,
    inputs: AnticipatoryDecisionInputs,
    policy: AnticipatoryDecisionPolicy,
) -> bool:
    return (
        inputs.impact_score
        >= policy.watch_impact_threshold
        or inputs.trajectory_rising
        or inputs.forecast_worsening
        or (
            inputs.cascade_intensity_score
            >= policy.watch_impact_threshold
        )
    )


def assess_anticipatory_decision(
    *,
    inputs: AnticipatoryDecisionInputs,
    policy: AnticipatoryDecisionPolicy | None = None,
) -> AnticipatoryDecision:
    """
    Convert PRAVAHA's analytical outputs into one operational
    decision-support state.

    Safety rules:

    - Prediction does not become authority confirmation.
    - EVACUATION_SUPPORT does not mean an evacuation order.
    - ACTIVE_EVENT requires observation or authority confirmation.
    - Insufficient confidence does not become NORMAL.
    """

    if policy is None:
        policy = AnticipatoryDecisionPolicy()

    _validate_policy(
        policy
    )

    confidence = inputs.system_confidence

    confidence_level = _confidence_level(
        confidence,
        policy,
    )

    margin = _clearance_margin(
        inputs
    )

    reasons: list[str] = []

    if inputs.impact_score >= policy.high_impact_threshold:
        reasons.append(
            "current_local_impact_high"
        )

    elif (
        inputs.impact_score
        >= policy.warning_impact_threshold
    ):
        reasons.append(
            "current_local_impact_warning"
        )

    elif (
        inputs.impact_score
        >= policy.watch_impact_threshold
    ):
        reasons.append(
            "current_local_impact_elevated"
        )

    if inputs.trajectory_rapidly_rising:
        reasons.append(
            "risk_trajectory_rapidly_rising"
        )

    elif inputs.trajectory_rising:
        reasons.append(
            "risk_trajectory_rising"
        )

    if (
        inputs.projected_high_threshold_minutes
        is not None
    ):
        reasons.append(
            "high_risk_threshold_projection_available"
        )

        if (
            inputs.projected_high_threshold_minutes
            <= policy.evacuation_support_horizon_minutes
        ):
            reasons.append(
                "high_risk_threshold_approaching_soon"
            )

    if inputs.forecast_worsening:
        reasons.append(
            "forecast_scenarios_indicate_worsening"
        )

    if (
        inputs.forecast_worst_consequence_score
        >= policy.severe_forecast_threshold
    ):
        reasons.append(
            "severe_forecast_consequence_possible"
        )

    if (
        inputs.cascade_intensity_score
        >= policy.significant_cascade_threshold
    ):
        reasons.append(
            "significant_multi_hazard_cascade"
        )

    if not inputs.evacuation_ready:
        reasons.append(
            "evacuation_readiness_degraded"
        )

    if margin is not None:
        if margin < 0.0:
            reasons.append(
                "evacuation_clearance_exceeds_hazard_window"
            )

        elif (
            margin
            <= policy.tight_clearance_margin_minutes
        ):
            reasons.append(
                "evacuation_clearance_margin_tight"
            )

        else:
            reasons.append(
                "evacuation_clearance_margin_available"
            )

    else:
        reasons.append(
            "evacuation_clearance_margin_unknown"
        )

    if (
        inputs.authority_event_status
        != AuthorityEventStatus.NONE
    ):
        reasons.append(
            "authority_status_present"
        )

    if inputs.event_observed:
        reasons.append(
            "hazard_event_observed"
        )

    if (
        confidence
        < policy.minimum_reliable_confidence
    ):
        reasons.append(
            "decision_confidence_insufficient"
        )

        return AnticipatoryDecision(
            unit_id=inputs.unit_id,
            operational_state=(
                OperationalState.INSUFFICIENT_DATA
            ),
            confidence=float(
                confidence
            ),
            confidence_level=confidence_level,
            evacuation_margin_minutes=margin,
            authority_event_status=(
                inputs.authority_event_status
            ),
            recommendation_code=(
                "VERIFY_DATA_AND_MAINTAIN_CAUTION"
            ),
            reasons=tuple(
                reasons
            ),
        )

    if _active_event_present(
        inputs
    ):
        return AnticipatoryDecision(
            unit_id=inputs.unit_id,
            operational_state=(
                OperationalState.ACTIVE_EVENT
            ),
            confidence=float(
                confidence
            ),
            confidence_level=confidence_level,
            evacuation_margin_minutes=margin,
            authority_event_status=(
                inputs.authority_event_status
            ),
            recommendation_code=(
                "SUPPORT_ACTIVE_RESPONSE"
            ),
            reasons=tuple(
                reasons
            ),
        )

    if _evacuation_support_needed(
        inputs=inputs,
        margin=margin,
        policy=policy,
    ):
        reasons.append(
            "anticipatory_evacuation_support_conditions_met"
        )

        return AnticipatoryDecision(
            unit_id=inputs.unit_id,
            operational_state=(
                OperationalState.EVACUATION_SUPPORT
            ),
            confidence=float(
                confidence
            ),
            confidence_level=confidence_level,
            evacuation_margin_minutes=margin,
            authority_event_status=(
                inputs.authority_event_status
            ),
            recommendation_code=(
                "PREPARE_EVACUATION_SUPPORT"
            ),
            reasons=tuple(
                reasons
            ),
        )

    if _warning_needed(
        inputs=inputs,
        policy=policy,
    ):
        return AnticipatoryDecision(
            unit_id=inputs.unit_id,
            operational_state=(
                OperationalState.WARNING
            ),
            confidence=float(
                confidence
            ),
            confidence_level=confidence_level,
            evacuation_margin_minutes=margin,
            authority_event_status=(
                inputs.authority_event_status
            ),
            recommendation_code=(
                "ESCALATE_MONITORING_AND_RESPONSE"
            ),
            reasons=tuple(
                reasons
            ),
        )

    if _prepare_needed(
        inputs=inputs,
        policy=policy,
    ):
        return AnticipatoryDecision(
            unit_id=inputs.unit_id,
            operational_state=(
                OperationalState.PREPARE
            ),
            confidence=float(
                confidence
            ),
            confidence_level=confidence_level,
            evacuation_margin_minutes=margin,
            authority_event_status=(
                inputs.authority_event_status
            ),
            recommendation_code=(
                "BEGIN_PREPARATORY_ACTIONS"
            ),
            reasons=tuple(
                reasons
            ),
        )

    if _watch_needed(
        inputs=inputs,
        policy=policy,
    ):
        return AnticipatoryDecision(
            unit_id=inputs.unit_id,
            operational_state=(
                OperationalState.WATCH
            ),
            confidence=float(
                confidence
            ),
            confidence_level=confidence_level,
            evacuation_margin_minutes=margin,
            authority_event_status=(
                inputs.authority_event_status
            ),
            recommendation_code=(
                "CONTINUE_ENHANCED_MONITORING"
            ),
            reasons=tuple(
                reasons
            ),
        )

    return AnticipatoryDecision(
        unit_id=inputs.unit_id,
        operational_state=(
            OperationalState.NORMAL
        ),
        confidence=float(
            confidence
        ),
        confidence_level=confidence_level,
        evacuation_margin_minutes=margin,
        authority_event_status=(
            inputs.authority_event_status
        ),
        recommendation_code=(
            "CONTINUE_ROUTINE_MONITORING"
        ),
        reasons=tuple(
            reasons
        ),
    )