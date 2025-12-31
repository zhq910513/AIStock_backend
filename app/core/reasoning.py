from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings


@dataclass
class ReasoningOutput:
    decision: str  # BUY/SELL/HOLD
    confidence: float
    reason_code: str
    params: dict


class ReasoningEngine:
    """
    推理大脑：这里提供“可替换”的最小实现。
    实盘应把特征提取、模型推理、realtime_equivalent 判断与血统绑定做全。
    """
    def infer(self, features: dict[str, Any]) -> ReasoningOutput:
        # Minimal deterministic logic
        price = float(features.get("price", 0.0))
        if price <= 0:
            return ReasoningOutput("HOLD", 0.0, "RC_NO_PRICE_V1", {"price": price})

        # Dumb rule: always HOLD in skeleton
        return ReasoningOutput("HOLD", 0.5, "RC_HOLD_BASELINE_V1", {"price": price})

    def versions(self) -> dict:
        return {
            "RuleSetVersionHash": settings.RULESET_VERSION_HASH,
            "ModelSnapshotUUID": settings.MODEL_SNAPSHOT_UUID,
            "StrategyContractHash": settings.STRATEGY_CONTRACT_HASH,
            "FeatureExtractorVersion": settings.FEATURE_EXTRACTOR_VERSION,
            "CostModelVersion": settings.COST_MODEL_VERSION,
        }
