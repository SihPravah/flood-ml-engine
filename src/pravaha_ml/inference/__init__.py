from typing import Any


_ADAPTER_EXPORTS = {
    "DataStatus",
    "FusedCatchmentStateV21",
    "FusedWindowQuality",
    "adapt_fused_catchment_state",
}

_PREDICTOR_EXPORTS = {
    "DevelopmentRiskModel",
    "DrainageContext",
    "EvacuationContext",
    "IntegratedIntelligence",
    "ModelProvider",
    "ModelRuntimeStatus",
    "PredictorContext",
    "RoadContext",
    "RouteDomainResult",
    "RouteEdgeSpec",
    "RouteStatus",
    "StaticCatchmentContext",
    "predict_intelligence",
}

__all__ = [
    "DataStatus",
    "DevelopmentRiskModel",
    "DrainageContext",
    "EvacuationContext",
    "FusedCatchmentStateV21",
    "FusedWindowQuality",
    "IntegratedIntelligence",
    "ModelProvider",
    "ModelRuntimeStatus",
    "PredictorContext",
    "RoadContext",
    "RouteDomainResult",
    "RouteEdgeSpec",
    "RouteStatus",
    "StaticCatchmentContext",
    "adapt_fused_catchment_state",
    "predict_intelligence",
]


def __getattr__(name: str) -> Any:
    if name in _ADAPTER_EXPORTS:
        from pravaha_ml.inference import fused_state_adapter

        value = getattr(fused_state_adapter, name)
    elif name in _PREDICTOR_EXPORTS:
        from pravaha_ml.inference import predictor

        value = getattr(predictor, name)
    else:
        raise AttributeError(
            f"module 'pravaha_ml.inference' has no attribute {name!r}"
        )

    globals()[name] = value
    return value
