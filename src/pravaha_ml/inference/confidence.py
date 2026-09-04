from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
    TemporalQualityReport,
)


class InputProvenance(str, Enum):
    OBSERVED = "OBSERVED"
    ESTIMATED = "ESTIMATED"
    MISSING = "MISSING"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class PredictionDisposition(str, Enum):
    """
    Operational interpretation of prediction reliability.

    NORMAL:
        Prediction may be used normally by downstream
        decision-support logic.

    CAUTION:
        Prediction may be shown, but uncertainty must be
        clearly communicated.

    INSUFFICIENT_DATA:
        PRAVAHA should not interpret a low model score as
        evidence that the area is safe.
    """

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class ConfidencePolicy:
    """
    Development policy for combining multiple reliability signals.

    These weights are deliberately explicit rather than hidden
    inside the model.

    Final values must be calibrated against real historical
    operational data.

    Weights do not need to sum to 1; they are normalized during
    confidence calculation.
    """

    temporal_quality_weight: float = 0.30
    source_availability_weight: float = 0.25
    provenance_weight: float = 0.20
    model_agreement_weight: float = 0.25

    estimated_input_penalty: float = 0.30

    minimum_source_availability: float = 0.40

    high_confidence_threshold: float = 0.80
    moderate_confidence_threshold: float = 0.60
    minimum_usable_confidence: float = 0.40


@dataclass(frozen=True)
class ConfidenceAssessment:
    overall_confidence: float
    confidence_level: ConfidenceLevel
    disposition: PredictionDisposition

    temporal_quality_score: float
    source_availability_score: float
    provenance_score: float
    model_agreement_score: float

    estimated_input_fraction: float
    source_availability_fraction: float

    model_disagreement: float

    can_treat_low_risk_as_reliable: bool

    reasons: tuple[str, ...]


def _validate_fraction(
    value: float,
    field_name: str,
) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )


def _validate_policy(
    policy: ConfidencePolicy,
) -> None:
    weights = [
        policy.temporal_quality_weight,
        policy.source_availability_weight,
        policy.provenance_weight,
        policy.model_agreement_weight,
    ]

    if any(
        weight < 0.0
        for weight in weights
    ):
        raise ValueError(
            "Confidence weights cannot be negative."
        )

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one confidence weight must be positive."
        )

    _validate_fraction(
        policy.estimated_input_penalty,
        "estimated_input_penalty",
    )

    _validate_fraction(
        policy.minimum_source_availability,
        "minimum_source_availability",
    )

    _validate_fraction(
        policy.high_confidence_threshold,
        "high_confidence_threshold",
    )

    _validate_fraction(
        policy.moderate_confidence_threshold,
        "moderate_confidence_threshold",
    )

    _validate_fraction(
        policy.minimum_usable_confidence,
        "minimum_usable_confidence",
    )

    if not (
        policy.minimum_usable_confidence
        <= policy.moderate_confidence_threshold
        <= policy.high_confidence_threshold
    ):
        raise ValueError(
            "Confidence thresholds must satisfy: "
            "minimum_usable_confidence <= "
            "moderate_confidence_threshold <= "
            "high_confidence_threshold."
        )


def _temporal_quality_score(
    temporal_quality: TemporalQualityReport,
) -> float:
    if (
        temporal_quality.level
        == TemporalQualityLevel.GOOD
    ):
        return 1.0

    if (
        temporal_quality.level
        == TemporalQualityLevel.DEGRADED
    ):
        return 0.65

    return 0.0


def _provenance_score(
    provenances: Iterable[
        InputProvenance
    ],
    estimated_input_penalty: float,
) -> tuple[float, float]:
    provenances = list(
        provenances
    )

    if not provenances:
        return 0.0, 1.0

    observed_count = sum(
        provenance
        == InputProvenance.OBSERVED
        for provenance in provenances
    )

    estimated_count = sum(
        provenance
        == InputProvenance.ESTIMATED
        for provenance in provenances
    )

    missing_count = sum(
        provenance
        == InputProvenance.MISSING
        for provenance in provenances
    )

    total = len(
        provenances
    )

    estimated_fraction = (
        estimated_count
        / total
    )

    missing_fraction = (
        missing_count
        / total
    )

    observed_fraction = (
        observed_count
        / total
    )

    score = (
        observed_fraction
        + estimated_fraction
        * (
            1.0
            - estimated_input_penalty
        )
    )

    score *= (
        1.0
        - missing_fraction
    )

    return (
        float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        ),
        float(
            estimated_fraction
        ),
    )


def _model_agreement_score(
    model_scores: Iterable[float],
) -> tuple[float, float]:
    """
    Estimate ensemble agreement from model risk probabilities.

    Perfect agreement:
        disagreement = 0
        agreement score = 1

    Maximum possible disagreement:
        disagreement = 1
        agreement score = 0

    This is not statistical uncertainty calibration. It is an
    explicit development-time model-disagreement signal.
    """

    model_scores = list(
        model_scores
    )

    if not model_scores:
        return 0.0, 1.0

    for score in model_scores:
        _validate_fraction(
            score,
            "model score",
        )

    if len(model_scores) == 1:
        return 0.75, 0.25

    disagreement = (
        max(model_scores)
        - min(model_scores)
    )

    agreement = (
        1.0
        - disagreement
    )

    return (
        float(
            np.clip(
                agreement,
                0.0,
                1.0,
            )
        ),
        float(
            disagreement
        ),
    )


def _classify_confidence(
    confidence: float,
    policy: ConfidencePolicy,
) -> ConfidenceLevel:
    if (
        confidence
        < policy.minimum_usable_confidence
    ):
        return (
            ConfidenceLevel.INSUFFICIENT
        )

    if (
        confidence
        >= policy.high_confidence_threshold
    ):
        return ConfidenceLevel.HIGH

    if (
        confidence
        >= policy.moderate_confidence_threshold
    ):
        return ConfidenceLevel.MODERATE

    return ConfidenceLevel.LOW


def assess_prediction_confidence(
    *,
    temporal_quality: TemporalQualityReport,
    source_availability_fraction: float,
    input_provenances: Iterable[
        InputProvenance
    ],
    model_scores: Iterable[float],
    policy: ConfidencePolicy | None = None,
) -> ConfidenceAssessment:
    """
    Combine data-quality and model-agreement evidence into a
    transparent prediction-confidence assessment.

    This confidence score is deliberately separate from the
    flash-flood risk score.

    Example:

        risk_score = 0.20
        overall_confidence = 0.92

    means:
        evidence strongly supports relatively low risk.

    But:

        risk_score = 0.20
        overall_confidence = 0.28

    means:
        evidence is too weak to interpret the low score as safe.

    Safety principle:

        LOW RISK + LOW CONFIDENCE != SAFE
    """

    if policy is None:
        policy = ConfidencePolicy()

    _validate_policy(
        policy
    )

    _validate_fraction(
        source_availability_fraction,
        "source_availability_fraction",
    )

    reasons: list[str] = []

    temporal_score = (
        _temporal_quality_score(
            temporal_quality
        )
    )

    provenance_score, estimated_fraction = (
        _provenance_score(
            provenances=(
                input_provenances
            ),
            estimated_input_penalty=(
                policy.estimated_input_penalty
            ),
        )
    )

    agreement_score, disagreement = (
        _model_agreement_score(
            model_scores=model_scores
        )
    )

    if (
        temporal_quality.level
        == TemporalQualityLevel.DEGRADED
    ):
        reasons.append(
            "temporal_data_degraded"
        )

    if (
        temporal_quality.level
        == TemporalQualityLevel.UNUSABLE
    ):
        reasons.append(
            "temporal_data_unusable"
        )

    if (
        source_availability_fraction
        < 1.0
    ):
        reasons.append(
            "some_sources_unavailable"
        )

    if (
        source_availability_fraction
        < policy.minimum_source_availability
    ):
        reasons.append(
            "source_availability_below_minimum"
        )

    if estimated_fraction > 0.0:
        reasons.append(
            "estimated_inputs_present"
        )

    if provenance_score < 1.0:
        reasons.append(
            "input_provenance_not_fully_observed"
        )

    if disagreement >= 0.25:
        reasons.append(
            "models_disagree"
        )

    weights = np.asarray(
        [
            policy.temporal_quality_weight,
            policy.source_availability_weight,
            policy.provenance_weight,
            policy.model_agreement_weight,
        ],
        dtype=float,
    )

    component_scores = np.asarray(
        [
            temporal_score,
            source_availability_fraction,
            provenance_score,
            agreement_score,
        ],
        dtype=float,
    )

    overall_confidence = float(
        np.average(
            component_scores,
            weights=weights,
        )
    )

    hard_block = (
        not temporal_quality.can_predict
        or (
            source_availability_fraction
            < policy.minimum_source_availability
        )
    )

    if hard_block:
        overall_confidence = min(
            overall_confidence,
            policy.minimum_usable_confidence
            - 0.01,
        )

    overall_confidence = float(
        np.clip(
            overall_confidence,
            0.0,
            1.0,
        )
    )

    confidence_level = (
        _classify_confidence(
            confidence=overall_confidence,
            policy=policy,
        )
    )

    if hard_block:
        disposition = (
            PredictionDisposition.INSUFFICIENT_DATA
        )

    elif (
        confidence_level
        in {
            ConfidenceLevel.LOW,
            ConfidenceLevel.INSUFFICIENT,
        }
    ):
        disposition = (
            PredictionDisposition.CAUTION
        )

    else:
        disposition = (
            PredictionDisposition.NORMAL
        )

    can_treat_low_risk_as_reliable = (
        disposition
        == PredictionDisposition.NORMAL
        and temporal_quality.can_predict
        and source_availability_fraction
        >= policy.minimum_source_availability
    )

    if (
        confidence_level
        == ConfidenceLevel.LOW
    ):
        reasons.append(
            "low_overall_confidence"
        )

    if (
        confidence_level
        == ConfidenceLevel.INSUFFICIENT
    ):
        reasons.append(
            "insufficient_overall_confidence"
        )

    return ConfidenceAssessment(
        overall_confidence=overall_confidence,
        confidence_level=confidence_level,
        disposition=disposition,
        temporal_quality_score=temporal_score,
        source_availability_score=(
            source_availability_fraction
        ),
        provenance_score=provenance_score,
        model_agreement_score=agreement_score,
        estimated_input_fraction=(
            estimated_fraction
        ),
        source_availability_fraction=(
            source_availability_fraction
        ),
        model_disagreement=disagreement,
        can_treat_low_risk_as_reliable=(
            can_treat_low_risk_as_reliable
        ),
        reasons=tuple(
            reasons
        ),
    )