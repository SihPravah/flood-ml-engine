import pytest

from pravaha_ml.anticipation.forecast import (
    ForecastProvenance,
    ForecastRainfallPoint,
    ForecastScenario,
    ForecastScenarioType,
    RainfallForecast,
)
from pravaha_ml.anticipation.scenarios import (
    ScenarioConsequence,
    ScenarioGenerationPolicy,
    build_standard_forecast_scenarios,
    evaluate_forecast_scenarios,
)


def make_forecast() -> RainfallForecast:
    return RainfallForecast(
        forecast_id="FC_001",
        points=(
            ForecastRainfallPoint(
                horizon_minutes=30,
                cumulative_rainfall_mm=20.0,
            ),
            ForecastRainfallPoint(
                horizon_minutes=60,
                cumulative_rainfall_mm=40.0,
            ),
            ForecastRainfallPoint(
                horizon_minutes=120,
                cumulative_rainfall_mm=65.0,
            ),
        ),
        provenance=(
            ForecastProvenance.NWP_DERIVED
        ),
        forecast_confidence=0.80,
    )


def make_consequence(
    *,
    risk_score: float,
    runoff_mm: float,
    drain_utilization: float,
    overflow_count: int,
    roads_to_avoid: int,
    route_compromised: bool,
    confidence: float = 0.85,
) -> ScenarioConsequence:
    if risk_score >= 0.85:
        risk_level = "SEVERE"
    elif risk_score >= 0.70:
        risk_level = "HIGH"
    elif risk_score >= 0.50:
        risk_level = "WARNING"
    elif risk_score >= 0.30:
        risk_level = "WATCH"
    else:
        risk_level = "LOW"

    return ScenarioConsequence(
        catchment_risk_score=risk_score,
        catchment_risk_level=risk_level,
        estimated_runoff_mm=runoff_mm,
        peak_drain_utilization=(
            drain_utilization
        ),
        overflowing_drain_count=(
            overflow_count
        ),
        roads_to_avoid=roads_to_avoid,
        evacuation_route_compromised=(
            route_compromised
        ),
        consequence_confidence=confidence,
        reasons=(),
    )


def test_standard_builder_creates_three_scenarios():
    scenarios = build_standard_forecast_scenarios(
        forecast=make_forecast()
    )

    assert len(scenarios) == 3

    assert {
        scenario.scenario_type
        for scenario in scenarios
    } == {
        ForecastScenarioType.WEAKENS,
        ForecastScenarioType.CONTINUES,
        ForecastScenarioType.INTENSIFIES,
    }


def test_weakening_scenario_reduces_rainfall():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    weakens = next(
        scenario
        for scenario in scenarios
        if scenario.scenario_type
        == ForecastScenarioType.WEAKENS
    )

    assert (
        weakens.rainfall_points[-1]
        .cumulative_rainfall_mm
        <
        forecast.points[-1]
        .cumulative_rainfall_mm
    )


def test_continuation_preserves_baseline_rainfall():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    continues = next(
        scenario
        for scenario in scenarios
        if scenario.scenario_type
        == ForecastScenarioType.CONTINUES
    )

    assert (
        continues.rainfall_points[-1]
        .cumulative_rainfall_mm
        ==
        pytest.approx(
            forecast.points[-1]
            .cumulative_rainfall_mm
        )
    )


def test_intensification_increases_rainfall():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    intensifies = next(
        scenario
        for scenario in scenarios
        if scenario.scenario_type
        == ForecastScenarioType.INTENSIFIES
    )

    assert (
        intensifies.rainfall_points[-1]
        .cumulative_rainfall_mm
        >
        forecast.points[-1]
        .cumulative_rainfall_mm
    )


def test_generated_stress_scenarios_do_not_invent_probability():
    scenarios = build_standard_forecast_scenarios(
        forecast=make_forecast()
    )

    assert all(
        scenario.probability is None
        for scenario in scenarios
    )


def test_invalid_multiplier_order_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Scenario rainfall multipliers must be ordered"
        ),
    ):
        ScenarioGenerationPolicy(
            weakens_multiplier=1.20,
            continues_multiplier=1.00,
            intensifies_multiplier=1.50,
        )


def test_scenario_evaluator_is_called_for_each_scenario():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    calls = []

    def evaluator(
        scenario,
    ):
        calls.append(
            scenario.scenario_id
        )

        return make_consequence(
            risk_score=0.50,
            runoff_mm=20.0,
            drain_utilization=0.80,
            overflow_count=0,
            roads_to_avoid=0,
            route_compromised=False,
        )

    evaluate_forecast_scenarios(
        forecast=forecast,
        scenarios=scenarios,
        evaluator=evaluator,
    )

    assert len(calls) == 3


def test_worst_consequence_is_identified_separately():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    def evaluator(
        scenario,
    ):
        if (
            scenario.scenario_type
            == ForecastScenarioType.WEAKENS
        ):
            return make_consequence(
                risk_score=0.30,
                runoff_mm=10.0,
                drain_utilization=0.50,
                overflow_count=0,
                roads_to_avoid=0,
                route_compromised=False,
            )

        if (
            scenario.scenario_type
            == ForecastScenarioType.CONTINUES
        ):
            return make_consequence(
                risk_score=0.65,
                runoff_mm=30.0,
                drain_utilization=0.95,
                overflow_count=0,
                roads_to_avoid=1,
                route_compromised=False,
            )

        return make_consequence(
            risk_score=0.90,
            runoff_mm=55.0,
            drain_utilization=1.30,
            overflow_count=2,
            roads_to_avoid=4,
            route_compromised=True,
        )

    result = evaluate_forecast_scenarios(
        forecast=forecast,
        scenarios=scenarios,
        evaluator=evaluator,
    )

    assert (
        "RAIN_INTENSIFIES"
        in result.worst_consequence_scenario_id
    )


def test_combined_confidence_uses_weaker_source():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    def evaluator(
        scenario,
    ):
        return make_consequence(
            risk_score=0.50,
            runoff_mm=20.0,
            drain_utilization=0.80,
            overflow_count=0,
            roads_to_avoid=0,
            route_compromised=False,
            confidence=0.60,
        )

    result = evaluate_forecast_scenarios(
        forecast=forecast,
        scenarios=scenarios,
        evaluator=evaluator,
    )

    assert all(
        assessment.combined_confidence
        == pytest.approx(0.60)
        for assessment in result.scenarios
    )


def test_no_probability_means_no_most_likely_scenario():
    forecast = make_forecast()

    scenarios = build_standard_forecast_scenarios(
        forecast=forecast
    )

    result = evaluate_forecast_scenarios(
        forecast=forecast,
        scenarios=scenarios,
        evaluator=lambda scenario: (
            make_consequence(
                risk_score=0.40,
                runoff_mm=15.0,
                drain_utilization=0.70,
                overflow_count=0,
                roads_to_avoid=0,
                route_compromised=False,
            )
        ),
    )

    assert (
        result.highest_likelihood_scenario_id
        is None
    )


def test_explicit_probabilities_can_identify_most_likely():
    forecast = make_forecast()

    scenarios = (
        ForecastScenario(
            scenario_id="SC_LOW",
            scenario_type=(
                ForecastScenarioType.WEAKENS
            ),
            rainfall_points=forecast.points,
            provenance=forecast.provenance,
            forecast_confidence=0.80,
            probability=0.20,
            rainfall_multiplier=0.50,
        ),
        ForecastScenario(
            scenario_id="SC_MAIN",
            scenario_type=(
                ForecastScenarioType.CONTINUES
            ),
            rainfall_points=forecast.points,
            provenance=forecast.provenance,
            forecast_confidence=0.80,
            probability=0.60,
            rainfall_multiplier=1.00,
        ),
        ForecastScenario(
            scenario_id="SC_HIGH",
            scenario_type=(
                ForecastScenarioType.INTENSIFIES
            ),
            rainfall_points=forecast.points,
            provenance=forecast.provenance,
            forecast_confidence=0.80,
            probability=0.20,
            rainfall_multiplier=1.50,
        ),
    )

    result = evaluate_forecast_scenarios(
        forecast=forecast,
        scenarios=scenarios,
        evaluator=lambda scenario: (
            make_consequence(
                risk_score=0.50,
                runoff_mm=20.0,
                drain_utilization=0.80,
                overflow_count=0,
                roads_to_avoid=0,
                route_compromised=False,
            )
        ),
    )

    assert (
        result.highest_likelihood_scenario_id
        == "SC_MAIN"
    )


def test_duplicate_scenario_ids_rejected():
    forecast = make_forecast()

    scenario = ForecastScenario(
        scenario_id="SAME",
        scenario_type=(
            ForecastScenarioType.CUSTOM
        ),
        rainfall_points=forecast.points,
        provenance=forecast.provenance,
        forecast_confidence=0.80,
        probability=None,
        rainfall_multiplier=1.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Forecast scenario IDs must be unique"
        ),
    ):
        evaluate_forecast_scenarios(
            forecast=forecast,
            scenarios=(
                scenario,
                scenario,
            ),
            evaluator=lambda item: (
                make_consequence(
                    risk_score=0.50,
                    runoff_mm=20.0,
                    drain_utilization=0.80,
                    overflow_count=0,
                    roads_to_avoid=0,
                    route_compromised=False,
                )
            ),
        )


def test_empty_scenario_set_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "At least one forecast scenario is required"
        ),
    ):
        evaluate_forecast_scenarios(
            forecast=make_forecast(),
            scenarios=(),
            evaluator=lambda item: (
                make_consequence(
                    risk_score=0.50,
                    runoff_mm=20.0,
                    drain_utilization=0.80,
                    overflow_count=0,
                    roads_to_avoid=0,
                    route_compromised=False,
                )
            ),
        )


def test_invalid_consequence_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "consequence_confidence must be between 0 and 1"
        ),
    ):
        make_consequence(
            risk_score=0.50,
            runoff_mm=20.0,
            drain_utilization=0.80,
            overflow_count=0,
            roads_to_avoid=0,
            route_compromised=False,
            confidence=1.40,
        )