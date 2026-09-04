from datetime import (
    datetime,
    timezone,
)

import pytest

from pravaha_ml.city.models import (
    CatchmentIntelligence,
)
from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
    TemporalQualityReport,
)
from pravaha_ml.inference.confidence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    PredictionDisposition,
)


UTC = timezone.utc


def make_hydrology() -> HydrologyFeatures:
    return HydrologyFeatures(
        rain_15m_mm=10.0,
        rain_30m_mm=20.0,
        rain_1h_mm=30.0,
        rain_3h_mm=40.0,
        rain_6h_mm=50.0,
        rain_24h_mm=80.0,
        api_mm=35.0,
        soil_moisture_percentage=80.0,
        soil_saturation=0.80,
        moisture_condition="WET",
        base_curve_number=80.0,
        effective_curve_number=90.0,
        runoff_mm=18.0,
        runoff_ratio=0.60,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        concentration_time_minutes=25.0,
    )


def make_confidence() -> ConfidenceAssessment:
    return ConfidenceAssessment(
        overall_confidence=0.90,
        confidence_level=(
            ConfidenceLevel.HIGH
        ),
        disposition=(
            PredictionDisposition.NORMAL
        ),
        temporal_quality_score=1.0,
        source_availability_score=1.0,
        provenance_score=1.0,
        model_agreement_score=0.95,
        estimated_input_fraction=0.0,
        source_availability_fraction=1.0,
        model_disagreement=0.05,
        can_treat_low_risk_as_reliable=True,
        reasons=(),
    )


def test_valid_catchment_intelligence():
    catchment = CatchmentIntelligence(
        catchment_id="C_001",
        risk_score=0.80,
        risk_level="HIGH",
        confidence=make_confidence(),
        hydrology=make_hydrology(),
        affected_population=1200,
    )

    assert (
        catchment.catchment_id
        == "C_001"
    )


def test_invalid_catchment_risk_rejected():
    with pytest.raises(
        ValueError,
        match="risk_score must be between 0 and 1",
    ):
        CatchmentIntelligence(
            catchment_id="C_001",
            risk_score=1.50,
            risk_level="HIGH",
            confidence=make_confidence(),
            hydrology=make_hydrology(),
        )


def test_negative_population_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "affected_population cannot be negative"
        ),
    ):
        CatchmentIntelligence(
            catchment_id="C_001",
            risk_score=0.50,
            risk_level="WARNING",
            confidence=make_confidence(),
            hydrology=make_hydrology(),
            affected_population=-1,
        )