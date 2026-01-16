"""Model v2 recommendation engine.

Goal (configurable):
- Within holding window (default 3 trading days), probability of reaching target profit (default 5%~8%).

This module intentionally separates:
- Feature extraction (from normalized facts / candidates)
- Scoring (a lightweight baseline that can be replaced by a trained model later)
- Persistence with full lineage and evidence for UI traceability

IMPORTANT (compat bridge):
- The current HTTP APIs and orchestrator still persist into legacy tables
  (ModelDecision/DecisionEvidence) for UI endpoints.
- Therefore this module exports `generate_for_batch_v2` and `persist_decisions_v2`
  with the same signature as v1, implemented as a thin compatibility layer.

NOTE:
- The native v2 run/reco/evidence tables are present, but the API surface is not
  switched yet. When you switch, you can call `generate_for_batch_run_v2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from math import exp, log
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.config import settings
from app.database import models
from sqlalchemy import select

# LightGBM model runtime
import lightgbm as lgb
from app.core.model_training import (
    OBJ_TP5_3D,
    OBJ_TP8_3D,
    build_features_from_snapshot,
    predict_contrib,
    predict_proba,
)
from app.utils.symbols import normalize_symbol


def _next_day_yyyymmdd(td: str) -> str:
    dt = datetime.strptime(td, "%Y%m%d")
    return (dt + timedelta(days=1)).strftime("%Y%m%d")


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _safe_logit(p: Optional[float]) -> float:
    if p is None:
        return 0.0
    # clamp
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return log(p / (1.0 - p))


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class ScoredCandidate:
    symbol: str
    action: str
    score: float
    confidence: float
    p_hit_target: float
    expected_max_return: Optional[float]
    p_limit_up_next_day: Optional[float]
    signals: Dict[str, Any]
    evidence: List[Tuple[str, str, Dict[str, Any]]]

@dataclass
class CandidateView:
    """A minimal attribute-view for scoring.

    We keep this to decouple scoring from the upstream table schema.
    """

    symbol: str
    p_limit_up: float
    turnover_rate: float = 0.0
    order_amount: float = 0.0
    sum_market_value: float = 0.0
    is_again_limit: int = 0
    high_days: Any = None
    limit_up_type: str = ""
    change_rate: Any = None


@dataclass
class _MLContext:
    tp5_booster: Optional[lgb.Booster]
    tp5_features: list[str]
    tp8_booster: Optional[lgb.Booster]
    tp8_features: list[str]


def _load_active_ml_context(db: Session) -> _MLContext:
    def _load(obj: str) -> tuple[Optional[lgb.Booster], list[str]]:
        row = db.execute(
            select(models.ModelArtifact)
            .where(models.ModelArtifact.objective == obj, models.ModelArtifact.is_active == True)  # noqa: E712
            .order_by(models.ModelArtifact.trained_ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None, []
        try:
            b = lgb.Booster(model_str=row.artifact_text)
            fl = list(row.feature_list or [])
            return b, fl
        except Exception:
            return None, []

    tp5_b, tp5_f = _load(OBJ_TP5_3D)
    tp8_b, tp8_f = _load(OBJ_TP8_3D)
    return _MLContext(tp5_booster=tp5_b, tp5_features=tp5_f, tp8_booster=tp8_b, tp8_features=tp8_f)



def score_candidate(
    c: models.LabelingCandidate,
    target_low: float,
    target_high: float,
    holding_days: int,
) -> ScoredCandidate:
    """Baseline scorer.

    Uses a small set of fields available in current candidate pool.

    When more data is ingested (1m bars / orderbook / sector flow / sentiment),
    extend this function to consume `feat_intraday_cutoff` etc.
    """

    symbol = c.symbol

    p_limit_up = _to_float(c.p_limit_up, default=0.2) if hasattr(c, "p_limit_up") else 0.2
    turnover_rate = _to_float(getattr(c, "turnover_rate", 0.0), default=0.0)
    order_amount = _to_float(getattr(c, "order_amount", 0.0), default=0.0)
    mkt = _to_float(getattr(c, "sum_market_value", 0.0), default=0.0)
    is_again = _to_int(getattr(c, "is_again_limit", 0), default=0)

    # Parse high_days like "2天2板" -> 2
    high_days = 0
    try:
        v = getattr(c, "high_days", None)
        if v:
            high_days = int(str(v).split("天")[0])
    except Exception:
        high_days = 0

    limit_up_type = (getattr(c, "limit_up_type", "") or "").strip()
    is_yizi = 1 if ("一字" in limit_up_type) else 0

    # Heuristic log-features
    logit_p = _safe_logit(p_limit_up)
    log_amt = log(max(order_amount, 1.0))
    log_mkt = log(max(mkt, 1.0))

    # Score: probability of hitting target return within holding window
    x = (
        -0.8
        + 0.55 * logit_p
        + 0.28 * float(high_days)
        - 0.60 * float(is_yizi)
        - 0.08 * min(turnover_rate, 20.0)
        + 0.10 * (log_amt - 12.0)
        - 0.08 * (log_mkt - 22.0)
        + 0.18 * float(is_again)
    )
    p_hit = _sigmoid(x)

    # Expected max return (coarse). Keep bounded.
    exp_max = max(0.0, min(0.25, (p_hit * (target_high + 0.02))))

    # Map p_hit -> action
    if p_hit >= 0.70:
        action = "买入"
    elif p_hit >= 0.55:
        action = "观察"
    else:
        action = "忽略"

    score = round(100.0 * p_hit, 2)
    confidence = round(p_hit, 4)

    signals: Dict[str, Any] = {
        "p_hit_target": round(p_hit, 6),
        "target_return_low": target_low,
        "target_return_high": target_high,
        "holding_days": holding_days,
        "expected_max_return": round(exp_max, 6),
        "p_limit_up": round(p_limit_up, 6),
        "high_days": getattr(c, "high_days", None),
        "limit_up_type": getattr(c, "limit_up_type", None),
        "turnover_rate": turnover_rate,
        "order_amount": order_amount,
        "sum_market_value": mkt,
        "is_again_limit": is_again,
    }

    evidence: List[Tuple[str, str, Dict[str, Any]]] = []

    evidence.append(
        (
            "P_HIT_TARGET_3D",
            f"三日内达到{int(target_low * 100)}%+收益概率（p_hit_target={p_hit:.4f}）",
            {
                "p_hit_target": round(p_hit, 6),
                "holding_days": holding_days,
                "target_low": target_low,
                "target_high": target_high,
            },
        )
    )

    evidence.append(
        (
            "INPUT_P_LIMIT_UP",
            f"候选池给出的涨停概率（p_limit_up={p_limit_up:.4f}），仅作为特征之一",
            {"p_limit_up": round(p_limit_up, 6)},
        )
    )

    evidence.append(
        (
            "MOMENTUM_HIGH_DAYS",
            f"连板/高标强度：{getattr(c, 'high_days', None) or 'N/A'}",
            {"high_days": getattr(c, "high_days", None)},
        )
    )

    if limit_up_type:
        evidence.append(
            (
                "LIMITUP_TYPE",
                (
                    f"涨停类型：{limit_up_type}（一字板更难买入，模型会降低可交易性评分）"
                    if is_yizi
                    else f"涨停类型：{limit_up_type}"
                ),
                {"limit_up_type": limit_up_type},
            )
        )

    evidence.append(
        (
            "RAW_SNAPSHOT",
            "候选池原始字段快照（用于追溯）",
            {
                "turnover_rate": turnover_rate,
                "order_amount": order_amount,
                "sum_market_value": mkt,
                "is_again_limit": is_again,
                "change_rate": getattr(c, "change_rate", None),
            },
        )
    )

    return ScoredCandidate(
        symbol=symbol,
        action=action,
        score=score,
        confidence=confidence,
        p_hit_target=p_hit,
        expected_max_return=exp_max,
        p_limit_up_next_day=p_limit_up,
        signals=signals,
        evidence=evidence,
    )


def score_candidate_with_ml(
    c: models.LabelingCandidate,
    target_low: float,
    target_high: float,
    holding_days: int,
    ml: _MLContext,
) -> ScoredCandidate:
    """Score candidate using active LightGBM models when available.

    If either model is missing, fall back to the baseline scorer.
    """
    if ml.tp5_booster is None or ml.tp8_booster is None or not ml.tp5_features or not ml.tp8_features:
        return score_candidate(c, target_low=target_low, target_high=target_high, holding_days=holding_days)

    symbol = normalize_symbol(c.symbol)

    extra = getattr(c, "extra", None) or {}
    snapshot = {
        "p_limit_up": getattr(c, "p_limit_up", None),
        "high_days": extra.get("high_days") or extra.get("high_days_n") or extra.get("high_days_text"),
        "turnover_rate": extra.get("turnover_rate"),
        "order_amount": extra.get("order_amount") or extra.get("order_volume"),
        "sum_market_value": extra.get("sum_market_value") or extra.get("currency_value"),
        "is_again_limit": extra.get("is_again_limit"),
        "change_rate": extra.get("change_rate"),
        "limit_up_type": extra.get("limit_up_type"),
    }

    feats = build_features_from_snapshot(snapshot)
    p5 = predict_proba(ml.tp5_booster, ml.tp5_features, feats)
    p8 = predict_proba(ml.tp8_booster, ml.tp8_features, feats)

    # score/action mapping (contract-driven)
    score_raw = 100.0 * (0.6 * p8 + 0.4 * p5)

    # simple tradeability penalty (non-label, proxy-only)
    penalty = 0.0
    lut = str(snapshot.get("limit_up_type") or "").strip()
    if "一字" in lut:
        penalty += 12.0
    tr = _to_float(snapshot.get("turnover_rate"), default=0.0)
    if tr <= 0.2:
        penalty += 3.0
    final_score = round(max(0.0, score_raw - penalty), 2)

    if final_score >= 70.0:
        action = "买入"
    elif final_score >= 55.0:
        action = "观察"
    else:
        action = "忽略"

    confidence = round(float(max(p5, p8)), 6)
    exp_max = max(0.0, min(0.25, (0.6 * p8 + 0.4 * p5) * (target_high + 0.02)))

    signals: Dict[str, Any] = {
        "p_tp5_3d": round(p5, 6),
        "p_tp8_3d": round(p8, 6),
        "score": final_score,
        "penalty": round(penalty, 2),
        "target_return_low": target_low,
        "target_return_high": target_high,
        "holding_days": holding_days,
        "feature_schema_version": str(settings.FEATURE_EXTRACTOR_VERSION or "v1"),
    }

    evidence: List[Tuple[str, str, Dict[str, Any]]] = []
    evidence.append(
        (
            "MODEL_LGBM_TP5_TP8",
            "LightGBM 双头模型输出（以 Open(T+1) 入场、三日内能否达到 +5%/+8%）",
            {"p_tp5_3d": round(p5, 6), "p_tp8_3d": round(p8, 6), "score": final_score},
        )
    )

    # top factor contributions from TP8 head (more selective)
    try:
        contrib = predict_contrib(ml.tp8_booster, ml.tp8_features, feats)
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        evidence.append(
            (
                "TOP_FEATURE_CONTRIB",
                "本次决策变化的主要因子贡献（TP8 头，按绝对贡献排序）",
                {"top": [{"feature": k, "contrib": float(v), "value": float(feats.get(k, 0.0))} for k, v in top]},
            )
        )
    except Exception:
        pass

    evidence.append(("RAW_SNAPSHOT", "候选池原始字段快照（用于追溯）", snapshot))

    return ScoredCandidate(
        symbol=symbol,
        action=action,
        score=final_score,
        confidence=confidence,
        p_hit_target=float(0.6 * p8 + 0.4 * p5),
        expected_max_return=exp_max,
        p_limit_up_next_day=_to_float(getattr(c, "p_limit_up", None), default=None),
        signals=signals,
        evidence=evidence,
    )


# ---------------------------
# Native v2 persistence (NOT yet wired to APIs)
# ---------------------------

def generate_for_batch_run_v2(
    db: Session,
    batch_id: str,
    model_name: str = "target_return_3d",
    model_version: str = "v2-baseline",
    asof_ts: Optional[datetime] = None,
    target_low: Optional[float] = None,
    target_high: Optional[float] = None,
    holding_days: Optional[int] = None,
) -> str:
    """Generate recommendations for a committed pool batch into v2 tables.

    Returns: run_id

    This function is intentionally NOT used by current APIs yet.
    """

    batch = db.query(models.LimitupPoolBatch).filter(models.LimitupPoolBatch.batch_id == batch_id).first()
    if not batch:
        raise ValueError(f"batch_id not found: {batch_id}")

    decision_day = _next_day_yyyymmdd(batch.trading_day)

    if asof_ts is None:
        asof_ts = datetime.utcnow()

    if target_low is None:
        target_low = float(getattr(settings, "TARGET_RETURN_LOW", 0.05) or 0.05)
    if target_high is None:
        target_high = float(getattr(settings, "TARGET_RETURN_HIGH", 0.08) or 0.08)
    if holding_days is None:
        holding_days = int(getattr(settings, "HOLDING_DAYS", 3) or 3)

    run_id = f"runv2_{batch_id}_{int(asof_ts.timestamp())}"

    run = models.ModelRunV2(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        decision_day=decision_day,
        asof_ts=asof_ts,
        target_return_low=target_low,
        target_return_high=target_high,
        holding_days=holding_days,
        params={
            "source": "limitup_pool",
            "batch_id": batch.batch_id,
            "batch_trading_day": batch.trading_day,
            "raw_hash": batch.raw_hash,
        },
        label_version=None,
    )
    db.add(run)
    db.flush()

    rows = (
        db.query(models.LimitupCandidate)
        .filter(models.LimitupCandidate.batch_id == batch_id)
        .filter(models.LimitupCandidate.candidate_status != "DROPPED")
        .order_by(models.LimitupCandidate.id.asc())
        .all()
    )

    candidates: List[CandidateView] = []
    for c in rows:
        raw = dict(c.raw_json or {})
        candidates.append(
            CandidateView(
                symbol=c.symbol,
                p_limit_up=float(c.p_limit_up or 0.0),
                turnover_rate=_to_float(raw.get("turnover_rate"), default=0.0),
                order_amount=_to_float(raw.get("order_amount") or raw.get("order_volume"), default=0.0),
                sum_market_value=_to_float(raw.get("sum_market_value") or raw.get("currency_value"), default=0.0),
                is_again_limit=_to_int(raw.get("is_again_limit"), default=0),
                high_days=raw.get("high_days"),
                limit_up_type=str(raw.get("limit_up_type") or ""),
                change_rate=raw.get("change_rate"),
            )
        )

    scored: List[ScoredCandidate] = [
        score_candidate(c, target_low=target_low, target_high=target_high, holding_days=holding_days)
        for c in candidates
    ]

    scored.sort(key=lambda x: (x.p_hit_target, x.score), reverse=True)

    top_n = int(getattr(settings, "RECO_TOP_N", 10) or 10)
    kept = scored[:top_n]

    for item in kept:
        reco = models.ModelRecoV2(
            run_id=run_id,
            symbol=item.symbol,
            action=item.action,
            score=item.score,
            confidence=item.confidence,
            p_hit_target=float(item.p_hit_target),
            expected_max_return=item.expected_max_return,
            p_limit_up_next_day=item.p_limit_up_next_day,
            signals=item.signals,
            created_ts=asof_ts,
        )
        db.add(reco)
        db.flush()

        for reason_code, reason_text, fields in item.evidence:
            ev = models.ModelRecoEvidenceV2(
                reco_id=reco.reco_id,
                reason_code=reason_code,
                reason_text=reason_text,
                evidence_fields=fields,
                evidence_refs={"batch_id": batch_id, "symbol": item.symbol},
                created_ts=asof_ts,
            )
            db.add(ev)

    db.commit()
    return run_id


# ---------------------------
# Compatibility layer for current APIs/orchestrator
# ---------------------------

# Reuse v1 decision tables for now (ModelDecision / DecisionEvidence)
import uuid
from datetime import date, time

from app.core.recommender_v1 import EvidenceItem, RecommendationItem  # noqa: E402
from app.utils.crypto import sha256_hex  # noqa: E402
from app.utils.trading_calendar import next_n_trading_days  # noqa: E402
from app.utils.time import now_shanghai, SH_TZ  # noqa: E402


def generate_for_batch_v2(
    s: Session,
    batch: models.LimitupPoolBatch,
    topn: int | None = None,
) -> list[RecommendationItem]:
    """Generate recommendation items (API-compatible, but model-driven).

    This uses the active LightGBM artifacts (TP5/TP8) when present.
    It still outputs RecommendationItem so current UI endpoints keep working.
    """

    n = int(topn or settings.RECOMMEND_TOPN or 10)
    target_low = float(getattr(settings, "TARGET_RETURN_LOW", 0.05) or 0.05)
    target_high = float(getattr(settings, "TARGET_RETURN_HIGH", 0.08) or 0.08)
    holding_days = int(getattr(settings, "HOLDING_DAYS", 3) or 3)

    # Load active ML artifacts once
    ml = _load_active_ml_context(s)

    rows = (
        s.execute(
            select(models.LimitupCandidate)
            .where(models.LimitupCandidate.batch_id == batch.batch_id)
            .where(models.LimitupCandidate.candidate_status != "DROPPED")
        )
        .scalars()
        .all()
    )

    decision_day = _next_day_yyyymmdd(batch.trading_day)

    class _Cand:
        def __init__(self, symbol: str, p_limit_up: float | None, extra: dict | None):
            self.symbol = symbol
            self.p_limit_up = p_limit_up
            self.extra = extra or {}

    items: list[tuple[ScoredCandidate, models.LimitupCandidate]] = []
    for r in rows:
        cand = _Cand(symbol=r.symbol, p_limit_up=getattr(r, "p_limit_up", None), extra=dict(r.raw_json or {}))
        scored = score_candidate_with_ml(cand, target_low=target_low, target_high=target_high, holding_days=holding_days, ml=ml)
        items.append((scored, r))

    # Sort by score then confidence
    items.sort(key=lambda x: (float(x[0].score), float(x[0].confidence)), reverse=True)
    kept = items[:n]

    out: list[RecommendationItem] = []
    for scored, r in kept:
        act = scored.action
        if act in ("买入", "BUY"):
            action = "BUY"
        elif act in ("观察", "WATCH"):
            action = "WATCH"
        else:
            action = "AVOID"

        # deterministic decision_id
        decision_id = sha256_hex(
            f"{batch.trading_day}|{decision_day}|{normalize_symbol(r.symbol)}|{float(scored.score):.4f}|{action}".encode("utf-8")
        )[:64]

        refs_base = {
            "batch_id": batch.batch_id,
            "candidate_id": r.id,
            "candidate_symbol": r.symbol,
            "raw_hash": batch.raw_hash,
            "model": {
                "tp5_active": bool(ml.tp5_booster is not None),
                "tp8_active": bool(ml.tp8_booster is not None),
            },
        }

        evs: list[EvidenceItem] = []
        for reason_code, reason_text, fields in scored.evidence:
            evs.append(
                EvidenceItem(
                    reason_code=str(reason_code),
                    reason_text=str(reason_text),
                    evidence_fields=dict(fields or {}),
                    evidence_refs=refs_base,
                )
            )

        # pad evidence to >=3 for UI consistency
        while len(evs) < 3:
            evs.append(
                EvidenceItem(
                    reason_code="EVIDENCE_PADDING",
                    reason_text="证据不足（已按最小可追溯集输出）",
                    evidence_fields=scored.signals,
                    evidence_refs=refs_base,
                )
            )

        out.append(
            RecommendationItem(
                decision_id=decision_id,
                symbol=normalize_symbol(r.symbol),
                name=str(r.name or ""),
                action=action,
                score=float(scored.score),
                confidence=float(scored.confidence),
                evidence=evs,
            )
        )

    return out


def persist_decisions_v2(
    s: Session,
    batch: models.LimitupPoolBatch,
    items: list[RecommendationItem],
) -> None:
    """Persist decisions/evidence + write audit-first snapshot tables."""

    # 1) Legacy persistence (keeps current API endpoints stable)
    from app.core import recommender_v1 as _reco_v1  # local import to avoid cycles

    _reco_v1.persist_decisions(s, batch, items)

    # 2) Snapshot / Trajectory persistence (new schema)
    try:
        t_day = datetime.strptime(batch.trading_day, "%Y%m%d").date()
    except Exception:
        return

    cutoff_ts = datetime.combine(t_day, time(15, 30, 0), tzinfo=SH_TZ)
    gen_ts = now_shanghai()

    # Best-effort model_version for snapshots: tie to active artifact versions if present
    try:
        tp5_ver = s.execute(
            select(models.ModelArtifact.model_version)
            .where(models.ModelArtifact.objective == OBJ_TP5_3D, models.ModelArtifact.is_active == True)  # noqa: E712
            .order_by(models.ModelArtifact.trained_ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        tp8_ver = s.execute(
            select(models.ModelArtifact.model_version)
            .where(models.ModelArtifact.objective == OBJ_TP8_3D, models.ModelArtifact.is_active == True)  # noqa: E712
            .order_by(models.ModelArtifact.trained_ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        model_version = f"tp5:{tp5_ver or 'NA'}|tp8:{tp8_ver or 'NA'}"
    except Exception:
        model_version = "baseline"

    # compute sellable days with calendar (fallback weekend-only)
    try:
        t1, t2, t3 = next_n_trading_days(t_day, 3, session=s)
    except Exception:
        # fallback: naive
        t1, t2, t3 = t_day, t_day, t_day

    for it in items:
        inst = normalize_symbol(it.symbol)
        window_id: str | None = None
        if it.action == "BUY":
            # create or reuse window for the same (instrument, signal_day_T)
            existing = s.execute(
                select(models.TradeWindow).where(
                    models.TradeWindow.instrument_id == inst,
                    models.TradeWindow.signal_day_T == t_day,
                    models.TradeWindow.status == "OPEN",
                )
            ).scalar_one_or_none()
            if existing is None:
                window_id = str(uuid.uuid4())
                s.add(
                    models.TradeWindow(
                        window_id=window_id,
                        instrument_id=inst,
                        signal_day_T=t_day,
                        entry_ref_type="OPEN_T+1",
                        entry_ref_px=None,
                        sellable_start_day=t1,
                        sellable_end_day=t3,
                        status="OPEN",
                        close_reason=None,
                        created_ts=gen_ts,
                    )
                )
                s.flush()
            else:
                window_id = existing.window_id

        # pull p_tp5/p_tp8 from evidence if present
        p_tp5 = None
        p_tp8 = None
        score = None
        for ev in it.evidence:
            if ev.reason_code == "MODEL_LGBM_TP5_TP8":
                p_tp5 = ev.evidence_fields.get("p_tp5_3d")
                p_tp8 = ev.evidence_fields.get("p_tp8_3d")
                score = ev.evidence_fields.get("score")

        snapshot_id = str(uuid.uuid4())
        snap = models.ModelPredictionSnapshot(
            snapshot_id=snapshot_id,
            window_id=window_id,
            batch_id=str(batch.batch_id),
            instrument_id=inst,
            asof_day=t_day,
            cutoff_ts=cutoff_ts,
            generated_ts=gen_ts,
            action=str(it.action),
            score=Decimal(str(score)) if score is not None else Decimal(str(it.score)),
            p_tp5_3d=Decimal(str(p_tp5)) if p_tp5 is not None else None,
            p_tp8_3d=Decimal(str(p_tp8)) if p_tp8 is not None else None,
            p_tp5_nextday=None,
            p_tp8_nextday=None,
            expected_best_day=None,
            confidence=Decimal(str(it.confidence)),
            model_version=str(model_version),
            feature_schema_version=str(settings.FEATURE_EXTRACTOR_VERSION or "v1"),
            data_lineage={"batch_id": batch.batch_id, "raw_hash": batch.raw_hash},
            quality_flags={},
        )
        s.add(snap)
        s.flush()

        # Evidence mirror into new evidence table (best-effort)
        for ev in it.evidence:
            s.add(
                models.ModelSnapshotEvidence(
                    snapshot_id=snapshot_id,
                    reason_code=str(ev.reason_code),
                    reason_text=str(ev.reason_text),
                    evidence_payload=dict(ev.evidence_fields or {}),
                    refs=dict(ev.evidence_refs or {}),
                    importance=None,
                    created_ts=gen_ts,
                )
            )

        # Dependency: at least tie back to batch raw hash
        s.add(
            models.SnapshotDataDependency(
                snapshot_id=snapshot_id,
                dep_type="RAW_HASH",
                ref_table="limitup_pool_batches",
                ref_keys={"batch_id": batch.batch_id},
                time_range_start=None,
                time_range_end=None,
                raw_hash=batch.raw_hash,
                note="candidate pool source",
            )
        )

        # Delta: link to previous snapshot by instrument (same model_version)
        prev = s.execute(
            select(models.ModelPredictionSnapshot)
            .where(
                models.ModelPredictionSnapshot.instrument_id == inst,
                models.ModelPredictionSnapshot.model_version == str(model_version),
                models.ModelPredictionSnapshot.cutoff_ts < cutoff_ts,
            )
            .order_by(models.ModelPredictionSnapshot.cutoff_ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        if prev is not None:
            try:
                dp5 = (Decimal(str(p_tp5)) - Decimal(str(prev.p_tp5_3d))) if (p_tp5 is not None and prev.p_tp5_3d is not None) else None
                dp8 = (Decimal(str(p_tp8)) - Decimal(str(prev.p_tp8_3d))) if (p_tp8 is not None and prev.p_tp8_3d is not None) else None
                ds = (Decimal(str(score)) - Decimal(str(prev.score))) if (score is not None and prev.score is not None) else None
            except Exception:
                dp5 = dp8 = ds = None
            s.add(
                models.ModelSnapshotDelta(
                    snapshot_id=snapshot_id,
                    prev_snapshot_id=prev.snapshot_id,
                    delta_p_tp5=dp5,
                    delta_p_tp8=dp8,
                    delta_score=ds,
                    top_changed_factors=[],
                    n_new_raw_payloads=None,
                    summary_text=None,
                )
            )

    s.commit()
