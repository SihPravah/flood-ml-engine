from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from pravaha_ml.features.rainfall import (
    RainfallObservation,
)
from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
    TemporalQualityPolicy,
    evaluate_temporal_quality,
)
from pravaha_ml.inference.confidence import (
    ConfidenceLevel,
    ConfidencePolicy,
    InputProvenance,
    PredictionDisposition,
    assess_prediction_confidence,
)


UTC = timezone.utc


def make_good_quality_report():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=20)
            ),
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=10)
            ),
            rainfall_mm_per_hr=40.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=5)
            ),
            rainfall_mm_per_hr=50.0,
        ),
    ]

    return evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
    )


def make_degraded_quality_report():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    policy = TemporalQualityPolicy(
        max_gap_minutes=30.0,
        max_staleness_minutes=20.0,
        degraded_gap_fraction=0.70,
        degraded_staleness_fraction=0.70,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=35)
            ),
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=14)
            ),
            rainfall_mm_per_hr=40.0,
        ),
    ]

    return evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
        policy=policy,
    )


def make_unusable_quality_report():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=60)
            ),
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=45)
            ),
            rainfall_mm_per_hr=40.0,
        ),
    ]

    return evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
    )


def test_high_quality_inputs_produce_high_confidence():
    quality = make_good_quality_report()

    result = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.82,
            0.84,
        ],
    )

    assert (
        result.overall_confidence
        >= 0.80
    )

    assert (
        result.confidence_level
        == ConfidenceLevel.HIGH
    )

    assert (
        result.disposition
        == PredictionDisposition.NORMAL
    )

    assert (
        result.can_treat_low_risk_as_reliable
        is True
    )


def test_estimated_inputs_reduce_confidence():
    quality = make_good_quality_report()

    observed = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.80,
            0.82,
        ],
    )

    estimated = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.ESTIMATED,
            InputProvenance.ESTIMATED,
        ],
        model_scores=[
            0.80,
            0.82,
        ],
    )

    assert (
        estimated.overall_confidence
        < observed.overall_confidence
    )

    assert (
        "estimated_inputs_present"
        in estimated.reasons
    )


def test_missing_inputs_reduce_confidence_more_than_estimated():
    quality = make_good_quality_report()

    estimated = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=0.80,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.ESTIMATED,
        ],
        model_scores=[
            0.70,
            0.72,
        ],
    )

    missing = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=0.80,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.MISSING,
        ],
        model_scores=[
            0.70,
            0.72,
        ],
    )

    assert (
        missing.overall_confidence
        < estimated.overall_confidence
    )


def test_degraded_temporal_data_reduces_confidence():
    good_quality = make_good_quality_report()
    degraded_quality = (
        make_degraded_quality_report()
    )

    good = assess_prediction_confidence(
        temporal_quality=good_quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.70,
            0.72,
        ],
    )

    degraded = assess_prediction_confidence(
        temporal_quality=degraded_quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.70,
            0.72,
        ],
    )

    assert (
        degraded.overall_confidence
        < good.overall_confidence
    )

    assert (
        "temporal_data_degraded"
        in degraded.reasons
    )


def test_unusable_temporal_data_blocks_normal_disposition():
    quality = make_unusable_quality_report()

    assert (
        quality.level
        == TemporalQualityLevel.UNUSABLE
    )

    result = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.10,
            0.12,
        ],
    )

    assert (
        result.disposition
        == PredictionDisposition.INSUFFICIENT_DATA
    )

    assert (
        result.can_treat_low_risk_as_reliable
        is False
    )


def test_low_source_availability_blocks_normal_prediction():
    quality = make_good_quality_report()

    result = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=0.20,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.MISSING,
            InputProvenance.MISSING,
        ],
        model_scores=[
            0.15,
            0.18,
        ],
    )

    assert (
        result.disposition
        == PredictionDisposition.INSUFFICIENT_DATA
    )

    assert (
        result.can_treat_low_risk_as_reliable
        is False
    )

    assert (
        "source_availability_below_minimum"
        in result.reasons
    )


def test_model_disagreement_reduces_confidence():
    quality = make_good_quality_report()

    agreement = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.80,
            0.82,
        ],
    )

    disagreement = (
        assess_prediction_confidence(
            temporal_quality=quality,
            source_availability_fraction=1.0,
            input_provenances=[
                InputProvenance.OBSERVED,
                InputProvenance.OBSERVED,
            ],
            model_scores=[
                0.20,
                0.85,
            ],
        )
    )

    assert (
        disagreement.model_disagreement
        > agreement.model_disagreement
    )

    assert (
        disagreement.overall_confidence
        < agreement.overall_confidence
    )

    assert (
        "models_disagree"
        in disagreement.reasons
    )


def test_single_model_gets_reduced_agreement_score():
    quality = make_good_quality_report()

    result = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=1.0,
        input_provenances=[
            InputProvenance.OBSERVED,
        ],
        model_scores=[
            0.75,
        ],
    )

    assert (
        result.model_agreement_score
        == pytest.approx(0.75)
    )


def test_invalid_source_availability_rejected():
    quality = make_good_quality_report()

    with pytest.raises(
        ValueError,
        match=(
            "source_availability_fraction "
            "must be between 0 and 1"
        ),
    ):
        assess_prediction_confidence(
            temporal_quality=quality,
            source_availability_fraction=1.50,
            input_provenances=[
                InputProvenance.OBSERVED,
            ],
            model_scores=[
                0.70,
            ],
        )


def test_invalid_model_score_rejected():
    quality = make_good_quality_report()

    with pytest.raises(
        ValueError,
        match=(
            "model score must be between 0 and 1"
        ),
    ):
        assess_prediction_confidence(
            temporal_quality=quality,
            source_availability_fraction=1.0,
            input_provenances=[
                InputProvenance.OBSERVED,
            ],
            model_scores=[
                1.50,
            ],
        )


def test_invalid_confidence_policy_rejected():
    quality = make_good_quality_report()

    policy = ConfidencePolicy(
        temporal_quality_weight=-1.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Confidence weights cannot be negative"
        ),
    ):
        assess_prediction_confidence(
            temporal_quality=quality,
            source_availability_fraction=1.0,
            input_provenances=[
                InputProvenance.OBSERVED,
            ],
            model_scores=[
                0.70,
            ],
            policy=policy,
        )


def test_low_risk_cannot_be_called_reliable_when_confidence_is_insufficient():
    """
    Safety-critical invariant.

    Even if downstream model risk happens to be low, weak input
    evidence must prevent PRAVAHA from interpreting that result
    as confidently safe.
    """

    quality = make_good_quality_report()

    result = assess_prediction_confidence(
        temporal_quality=quality,
        source_availability_fraction=0.20,
        input_provenances=[
            InputProvenance.MISSING,
            InputProvenance.ESTIMATED,
            InputProvenance.MISSING,
        ],
        model_scores=[
            0.10,
            0.15,
        ],
    )

    assert (
        result.can_treat_low_risk_as_reliable
        is False
    )

    assert (
        result.disposition
        == PredictionDisposition.INSUFFICIENT_DATA
    )