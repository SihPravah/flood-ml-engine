import pytest

from pravaha_ml.models.risk import (
    RiskLevel,
    classify_risk,
)


def test_low_risk():
    assert classify_risk(0.10) == RiskLevel.LOW


def test_watch_risk():
    assert classify_risk(0.35) == RiskLevel.WATCH


def test_warning_risk():
    assert classify_risk(0.55) == RiskLevel.WARNING


def test_high_risk():
    assert classify_risk(0.75) == RiskLevel.HIGH


def test_severe_risk():
    assert classify_risk(0.90) == RiskLevel.SEVERE


def test_invalid_risk_score_rejected():
    with pytest.raises(
        ValueError,
        match="risk_score must be between 0 and 1",
    ):
        classify_risk(1.5)