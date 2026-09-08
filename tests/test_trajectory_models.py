from datetime import (
    datetime,
    timezone,
)

import pytest

from pravaha_ml.anticipation.models import (
    RiskObservation,
)


UTC = timezone.utc


def test_valid_risk_observation():
    observation = RiskObservation(
        timestamp=datetime(
            2026,
            9,
            8,
            12,
            0,
            tzinfo=UTC,
        ),
        risk_score=0.50,
    )

    assert (
        observation.risk_score
        == pytest.approx(0.50)
    )


def test_naive_timestamp_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "timestamp must be timezone-aware"
        ),
    ):
        RiskObservation(
            timestamp=datetime(
                2026,
                9,
                8,
                12,
                0,
            ),
            risk_score=0.50,
        )


def test_risk_above_one_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "risk_score must be between 0 and 1"
        ),
    ):
        RiskObservation(
            timestamp=datetime(
                2026,
                9,
                8,
                12,
                0,
                tzinfo=UTC,
            ),
            risk_score=1.20,
        )


def test_negative_risk_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "risk_score must be between 0 and 1"
        ),
    ):
        RiskObservation(
            timestamp=datetime(
                2026,
                9,
                8,
                12,
                0,
                tzinfo=UTC,
            ),
            risk_score=-0.10,
        )