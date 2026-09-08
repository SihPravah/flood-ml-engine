from dataclasses import dataclass
from typing import Callable

from pravaha_ml.anticipation.forecast import (
    ForecastRainfallPoint,
    ForecastScenario,
    ForecastScenarioType,
    RainfallForecast,
)


@dataclass(frozen=True)
class ScenarioGenerationPolicy:
    """
    Stress-test multipliers around a baseline rainfall forecast.

    These are development scenario assumptions.

    They are NOT automatically interpreted as calibrated forecast
    probabilities.

    Example:

        baseline forecast = 40 mm next hour

        weakens:
            0.50 * 40 = 20 mm

        continues:
            1.00 * 40 = 40 mm

        intensifies:
            1.50 * 40 = 60 mm

    In production these can later be replaced by actual ensemble
    forecast members.
    """

    weakens_multiplier: float = 0.50
    continues_multiplier: float = 1.00
    intensifies_multiplier: float = 1.50

    def __post_init__(self) -> None:
        values = {
            "weakens_multiplier": (
                self.weakens_multiplier
            ),
            "continues_multiplier": (
                self.continues_multiplier
            ),
            "intensifies_multiplier": (
                self.intensifies_multiplier
            ),
        }

        for field_name, value in values.items():
            if value < 0.0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if not (
            self.weakens_multiplier
            <= self.continues_multiplier
            <= self.intensifies_multiplier
        ):
            raise ValueError(
                "Scenario rainfall multipliers must be ordered."
            )


@dataclass(frozen=True)
class ScenarioConsequence:
    """
    Consequence returned after one rainfall scenario has passed
    through the PRAVAHA hazard pipeline.

    The scenario engine itself does NOT calculate these values.

    A supplied evaluator should eventually run:

        rainfall forcing
            -> hydrology
            -> runoff
            -> flash-flood model
            -> drainage network
            -> road risk

    This avoids arbitrary direct rainfall-to-risk arithmetic.
    """

    catchment_risk_score: float
    catchment_risk_level: str

    estimated_runoff_mm: float

    peak_drain_utilization: float

    overflowing_drain_count: int
    roads_to_avoid: int

    evacuation_route_compromised: bool

    consequence_confidence: float

    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.catchment_risk_score <= 1.0:
            raise ValueError(
                "catchment_risk_score must be between 0 and 1."
            )

        if not self.catchment_risk_level.strip():
            raise ValueError(
                "catchment_risk_level cannot be empty."
            )

        if self.estimated_runoff_mm < 0.0:
            raise ValueError(
                "estimated_runoff_mm cannot be negative."
            )

        if self.peak_drain_utilization < 0.0:
            raise ValueError(
                "peak_drain_utilization cannot be negative."
            )

        if self.overflowing_drain_count < 0:
            raise ValueError(
                "overflowing_drain_count cannot be negative."
            )

        if self.roads_to_avoid < 0:
            raise ValueError(
                "roads_to_avoid cannot be negative."
            )

        if not 0.0 <= self.consequence_confidence <= 1.0:
            raise ValueError(
                "consequence_confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class ScenarioAssessment:
    scenario: ForecastScenario

    consequence: ScenarioConsequence

    combined_confidence: float


@dataclass(frozen=True)
class ForecastScenarioAssessment:
    """
    Consolidated comparison of multiple future rainfall scenarios.
    """

    forecast_id: str

    scenarios: tuple[
        ScenarioAssessment,
        ...
    ]

    worst_consequence_scenario_id: str

    highest_likelihood_scenario_id: str | None


def _scale_points(
    *,
    forecast: RainfallForecast,
    multiplier: float,
) -> tuple[
    ForecastRainfallPoint,
    ...
]:
    return tuple(
        ForecastRainfallPoint(
            horizon_minutes=(
                point.horizon_minutes
            ),
            cumulative_rainfall_mm=(
                point.cumulative_rainfall_mm
                * multiplier
            ),
        )
        for point in forecast.points
    )


def build_standard_forecast_scenarios(
    *,
    forecast: RainfallForecast,
    policy: ScenarioGenerationPolicy | None = None,
) -> tuple[
    ForecastScenario,
    ...
]:
    """
    Construct three explicit stress-test scenarios from a baseline
    short-horizon rainfall forecast.

    IMPORTANT:

    These are NOT probabilities automatically assigned by PRAVAHA.

    If the baseline forecast contains a probability, that
    probability belongs to the baseline forecast, not automatically
    to all three stress scenarios.

    Therefore generated stress scenarios have probability=None.
    """

    if policy is None:
        policy = ScenarioGenerationPolicy()

    scenario_specs = (
        (
            ForecastScenarioType.WEAKENS,
            policy.weakens_multiplier,
        ),
        (
            ForecastScenarioType.CONTINUES,
            policy.continues_multiplier,
        ),
        (
            ForecastScenarioType.INTENSIFIES,
            policy.intensifies_multiplier,
        ),
    )

    scenarios = []

    for (
        scenario_type,
        multiplier,
    ) in scenario_specs:
        scenarios.append(
            ForecastScenario(
                scenario_id=(
                    f"{forecast.forecast_id}_"
                    f"{scenario_type.value}"
                ),
                scenario_type=scenario_type,
                rainfall_points=_scale_points(
                    forecast=forecast,
                    multiplier=multiplier,
                ),
                provenance=forecast.provenance,
                forecast_confidence=(
                    forecast.forecast_confidence
                ),
                probability=None,
                rainfall_multiplier=multiplier,
            )
        )

    return tuple(
        scenarios
    )


def _consequence_severity_key(
    assessment: ScenarioAssessment,
) -> tuple[
    float,
    int,
    int,
    float,
]:
    """
    Transparent ordering for identifying the most damaging modeled
    scenario.

    Priority uses:
        catchment risk
        roads to avoid
        overflowing drains
        drain utilization

    This is consequence ordering, NOT scenario probability.
    """

    consequence = assessment.consequence

    return (
        consequence.catchment_risk_score,
        consequence.roads_to_avoid,
        consequence.overflowing_drain_count,
        consequence.peak_drain_utilization,
    )


def evaluate_forecast_scenarios(
    *,
    forecast: RainfallForecast,
    scenarios: tuple[
        ForecastScenario,
        ...
    ],
    evaluator: Callable[
        [ForecastScenario],
        ScenarioConsequence,
    ],
) -> ForecastScenarioAssessment:
    """
    Evaluate future rainfall scenarios using an externally supplied
    PRAVAHA consequence evaluator.

    This architecture is intentional.

    The scenario layer does NOT contain a formula like:

        rainfall * arbitrary_weight = flood risk

    Instead, the evaluator can later invoke the real physical/model
    pipeline.

    Combined confidence is conservatively limited by the weaker of:

        rainfall forecast confidence
        consequence-model confidence
    """

    if not scenarios:
        raise ValueError(
            "At least one forecast scenario is required."
        )

    scenario_ids = [
        scenario.scenario_id
        for scenario in scenarios
    ]

    if len(
        scenario_ids
    ) != len(
        set(scenario_ids)
    ):
        raise ValueError(
            "Forecast scenario IDs must be unique."
        )

    assessments: list[
        ScenarioAssessment
    ] = []

    for scenario in scenarios:
        consequence = evaluator(
            scenario
        )

        combined_confidence = min(
            scenario.forecast_confidence,
            consequence.consequence_confidence,
        )

        assessments.append(
            ScenarioAssessment(
                scenario=scenario,
                consequence=consequence,
                combined_confidence=float(
                    combined_confidence
                ),
            )
        )

    worst = max(
        assessments,
        key=_consequence_severity_key,
    )

    probability_assessments = [
        assessment
        for assessment in assessments
        if assessment.scenario.probability
        is not None
    ]

    highest_likelihood_scenario_id = None

    if probability_assessments:
        most_likely = max(
            probability_assessments,
            key=lambda assessment: (
                assessment.scenario.probability
            ),
        )

        highest_likelihood_scenario_id = (
            most_likely.scenario.scenario_id
        )

    return ForecastScenarioAssessment(
        forecast_id=forecast.forecast_id,
        scenarios=tuple(
            assessments
        ),
        worst_consequence_scenario_id=(
            worst.scenario.scenario_id
        ),
        highest_likelihood_scenario_id=(
            highest_likelihood_scenario_id
        ),
    )