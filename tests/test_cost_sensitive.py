import pytest

from pravaha_ml.models.baseline import (
    BaselineRiskModel,
)
from pravaha_ml.models.xgboost_model import (
    XGBoostRiskModel,
)
from pravaha_ml.training.cost_sensitive import (
    run_cost_sensitive_comparison,
)
from pravaha_ml.training.realistic_synthetic import (
    generate_realistic_synthetic_data,
)


def test_logistic_accepts_balanced_class_weight():
    model = BaselineRiskModel(
        class_weight="balanced",
    )

    assert (
        model.class_weight
        == "balanced"
    )


def test_xgboost_accepts_positive_class_weight():
    model = XGBoostRiskModel(
        scale_pos_weight=2.0,
    )

    assert (
        model.scale_pos_weight
        == 2.0
    )


def test_invalid_xgboost_positive_weight_rejected():
    with pytest.raises(
        ValueError,
        match="scale_pos_weight must be greater than 0",
    ):
        XGBoostRiskModel(
            scale_pos_weight=0.0,
        )


def test_cost_sensitive_comparison_runs():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = (
        run_cost_sensitive_comparison(
            samples=samples,
            minimum_recall=0.80,
            random_state=42,
        )
    )

    assert result.train_size == 600
    assert result.calibration_size == 200
    assert result.test_size == 200


def test_all_model_results_have_valid_metrics():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=1000,
            random_state=42,
        )
    )

    result = (
        run_cost_sensitive_comparison(
            samples=samples,
            minimum_recall=0.80,
            random_state=42,
        )
    )

    model_results = [
        result.logistic_standard,
        result.logistic_balanced,
        result.xgboost_standard,
        result.xgboost_balanced,
    ]

    for model_result in model_results:
        metrics = (
            model_result.test_metrics
        )

        assert (
            0.0
            <= model_result.selected_threshold
            <= 1.0
        )

        assert (
            0.0
            <= metrics.precision
            <= 1.0
        )

        assert (
            0.0
            <= metrics.recall
            <= 1.0
        )

        assert (
            0.0
            <= metrics.specificity
            <= 1.0
        )

        assert (
            0.0
            <= metrics.false_positive_rate
            <= 1.0
        )

        assert (
            0.0
            <= metrics.f1
            <= 1.0
        )


def test_balanced_configurations_are_identified():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=500,
            random_state=42,
        )
    )

    result = (
        run_cost_sensitive_comparison(
            samples=samples,
            minimum_recall=0.80,
            random_state=42,
        )
    )

    assert (
        result.logistic_balanced.configuration
        == "balanced"
    )

    assert (
        result.xgboost_balanced.configuration
        == "balanced"
    )


def test_small_dataset_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=50,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="At least 100 samples",
    ):
        run_cost_sensitive_comparison(
            samples=samples
        )


def test_invalid_minimum_recall_rejected():
    samples = (
        generate_realistic_synthetic_data(
            n_samples=200,
            random_state=42,
        )
    )

    with pytest.raises(
        ValueError,
        match="minimum_recall",
    ):
        run_cost_sensitive_comparison(
            samples=samples,
            minimum_recall=1.20,
        )