from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    WARNING = "WARNING"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


def classify_risk(
    risk_score: float,
) -> RiskLevel:
    """
    Convert risk probability into an operational risk level.

    Important:
    These are development thresholds.

    They must later be calibrated using validation data and
    operational requirements.
    """

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            "risk_score must be between 0 and 1."
        )

    if risk_score >= 0.85:
        return RiskLevel.SEVERE

    if risk_score >= 0.70:
        return RiskLevel.HIGH

    if risk_score >= 0.50:
        return RiskLevel.WARNING

    if risk_score >= 0.30:
        return RiskLevel.WATCH

    return RiskLevel.LOW