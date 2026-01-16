from __future__ import annotations

"""Model training/inference (LightGBM).

This module enforces the project's trading contract:
  - Entry: Open(T+1)
  - Holding window: T+1..T+3 (3 trading days)
  - Labels: whether max High in window reaches +5%/+8% from entry

Two binary objectives:
  - TP5_3D -> p_tp5_3d
  - TP8_3D -> p_tp8_3d

Artifacts are stored in DB table `model_artifact` as LightGBM Booster model strings.
"""

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Iterable, Optional

import numpy as np
import lightgbm as lgb

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models
from app.utils.crypto import sha256_hex
from app.utils.symbols import normalize_symbol
from app.utils.time import SH_TZ
from app.core.label_tp3d import upsert_label_3d


OBJ_TP5_3D = "TP5_3D"
OBJ_TP8_3D = "TP8_3D"


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _parse_high_days(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v)
    import re

    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_logit(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return float(math.log(p / (1.0 - p)))


def build_features_from_snapshot(snapshot: dict[str, Any]) -> dict[str, float]:
    """Stable feature vector builder.

    Current available features are limited to candidate pool fields.
    The feature schema is versioned in settings.FEATURE_EXTRACTOR_VERSION.

    Missing values are encoded as 0 with *_missing flags.
    """
    f: dict[str, float] = {}

    p = _to_float(snapshot.get("p_limit_up"))
    f["p_limit_up"] = float(p or 0.0)
    f["p_limit_up_missing"] = 1.0 if p is None else 0.0

    hd = _parse_high_days(snapshot.get("high_days"))
    f["high_days_n"] = float(hd or 0.0)
    f["high_days_missing"] = 1.0 if hd is None else 0.0

    tr = _to_float(snapshot.get("turnover_rate"))
    f["turnover_rate"] = float(tr or 0.0)
    f["turnover_missing"] = 1.0 if tr is None else 0.0

    oa = _to_float(snapshot.get("order_amount") or snapshot.get("order_volume"))
    f["order_amount"] = float(oa or 0.0)
    f["order_amount_missing"] = 1.0 if oa is None else 0.0

    mv = _to_float(snapshot.get("sum_market_value") or snapshot.get("currency_value"))
    f["sum_market_value"] = float(mv or 0.0)
    f["sum_market_value_missing"] = 1.0 if mv is None else 0.0

    ial = _to_float(snapshot.get("is_again_limit"))
    f["is_again_limit"] = float(ial or 0.0)
    f["is_again_limit_missing"] = 1.0 if ial is None else 0.0

    cr = _to_float(snapshot.get("change_rate"))
    f["change_rate"] = float(cr or 0.0)
    f["change_rate_missing"] = 1.0 if cr is None else 0.0

    # limitup type one-hot (tiny)
    lut = str(snapshot.get("limit_up_type") or "").strip()
    for t in ("一字板", "换手板"):
        f[f"limit_up_type__{t}"] = 1.0 if lut == t else 0.0
    f["limit_up_type__OTHER"] = 1.0 if (lut and lut not in ("一字板", "换手板")) else 0.0
    f["limit_up_type_missing"] = 1.0 if not lut else 0.0

    # gentle nonlinearities
    f["log_order_amount"] = math.log1p(max(f["order_amount"], 0.0))
    f["log_sum_market_value"] = math.log1p(max(f["sum_market_value"], 0.0))

    # light interactions
    f["p_x_high_days"] = f["p_limit_up"] * f["high_days_n"]
    f["p_x_turnover"] = f["p_limit_up"] * f["turnover_rate"]
    f["logit_p_limit_up"] = _safe_logit(max(1e-6, min(1.0 - 1e-6, f["p_limit_up"])))

    return f


def _parse_yyyymmdd(td: str) -> date:
    return datetime.strptime(td, "%Y%m%d").date()


def _default_cutoff_ts(signal_day_T: date) -> datetime:
    # Default to the project's EOD+refine checkpoint (15:30)
    return datetime.combine(signal_day_T, time(15, 30, 0), tzinfo=SH_TZ)


def _candidate_snapshot(c: models.LabelingCandidate) -> dict[str, Any]:
    extra = c.extra or {}
    return {
        "p_limit_up": getattr(c, "p_limit_up", None),
        "high_days": extra.get("high_days") or extra.get("high_days_n") or extra.get("high_days_text"),
        "turnover_rate": extra.get("turnover_rate"),
        "order_amount": extra.get("order_amount") or extra.get("order_volume"),
        "sum_market_value": extra.get("sum_market_value") or extra.get("currency_value"),
        "is_again_limit": extra.get("is_again_limit"),
        "change_rate": extra.get("change_rate"),
        "limit_up_type": extra.get("limit_up_type"),
    }


@dataclass
class TrainResult:
    objective: str
    model_version: str
    metrics: dict[str, Any]
    feature_list: list[str]
    artifact_sha256: str


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    # fast AUC via rank; returns None if not computable
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_pos = ranks[pos].sum()
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    u = sum_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def ensure_labels_for_candidates(
    session: Session,
    trading_day: str,
    label_version: str | None = None,
    max_symbols: int = 5000,
) -> int:
    """Create ModelTrainingLabel3D rows for candidates on trading_day.

    This requires Daily OHLCV data to be present for T, T+1..T+3.
    Missing data -> symbol skipped.
    """
    td_date = _parse_yyyymmdd(trading_day)
    cutoff_ts = _default_cutoff_ts(td_date)
    lv = (label_version or settings.LABEL3D_VERSION or "v1").strip()

    q = (
        select(models.LabelingCandidate)
        .where(models.LabelingCandidate.trading_day == trading_day)
        .limit(max_symbols)
    )
    rows = session.execute(q).scalars().all()
    ok = 0
    for c in rows:
        inst = normalize_symbol(c.symbol)
        try:
            upsert_label_3d(session, inst, td_date, cutoff_ts, label_version=lv)
            ok += 1
        except Exception:
            # missing daily data or calendar; skip
            continue
    return ok


def train_objective_lightgbm(
    session: Session,
    objective: str,
    label_version: str | None = None,
    model_version: str | None = None,
    max_rows: int = 5000,
    valid_ratio: float = 0.2,
    seed: int | None = None,
) -> TrainResult:
    """Train a LightGBM binary classifier for the given objective.

    Training data sources:
      - features: LabelingCandidate (candidate pool) + extra fields
      - labels: ModelTrainingLabel3D (computed from Daily OHLCV)
    """
    obj = objective.strip().upper()
    if obj not in (OBJ_TP5_3D, OBJ_TP8_3D):
        raise ValueError(f"unknown objective: {objective}")

    lv = (label_version or settings.LABEL3D_VERSION or "v1").strip()
    mv = (model_version or f"lgbm-{obj.lower()}-{datetime.now(tz=SH_TZ).strftime('%Y%m%d%H%M%S')}").strip()
    rng = random.Random(seed if seed is not None else settings.LGBM_SEED)

    # Join labels with candidates (by day+symbol); instrument_id is normalized symbol
    # Label signal_day_T is Date; candidate trading_day is YYYYMMDD string.
    # We do a two-step fetch to keep SQLite compatibility and performance.
    label_q = (
        select(models.ModelTrainingLabel3D)
        .where(models.ModelTrainingLabel3D.label_version == lv)
        .order_by(models.ModelTrainingLabel3D.signal_day_T.desc())
        .limit(max_rows)
    )
    label_rows = session.execute(label_q).scalars().all()

    # Prefetch candidates for the involved days into a lookup map
    days = sorted({lab.signal_day_T.strftime("%Y%m%d") for lab in label_rows})
    cand_map: dict[tuple[str, str], models.LabelingCandidate] = {}
    if days:
        cands = session.execute(
            select(models.LabelingCandidate).where(models.LabelingCandidate.trading_day.in_(days))
        ).scalars().all()
        for c in cands:
            cand_map[(c.trading_day, normalize_symbol(c.symbol))] = c

    X_rows: list[dict[str, float]] = []
    y: list[int] = []

    for lab in label_rows:
        td = lab.signal_day_T.strftime("%Y%m%d")
        cand = cand_map.get((td, normalize_symbol(lab.instrument_id)))
        if cand is None:
            continue

        snap = _candidate_snapshot(cand)
        feats = build_features_from_snapshot(snap)

        label_val = bool(lab.label_tp5_3d) if obj == OBJ_TP5_3D else bool(lab.label_tp8_3d)

        X_rows.append(feats)
        y.append(1 if label_val else 0)

    if len(y) < 50:
        raise RuntimeError(f"not enough labeled samples to train: got {len(y)}")

    # feature list is stable: union of keys
    feature_list = sorted({k for r in X_rows for k in r.keys()})

    X = np.zeros((len(X_rows), len(feature_list)), dtype=np.float32)
    for i, r in enumerate(X_rows):
        for j, k in enumerate(feature_list):
            X[i, j] = float(r.get(k, 0.0))

    y_arr = np.asarray(y, dtype=np.int32)

    idx = list(range(len(y_arr)))
    rng.shuffle(idx)
    X = X[idx]
    y_arr = y_arr[idx]

    n_valid = int(max(1, round(len(y_arr) * valid_ratio)))
    n_train = len(y_arr) - n_valid
    if n_train < 10:
        raise RuntimeError("training split too small")

    X_train, y_train = X[:n_train], y_arr[:n_train]
    X_valid, y_valid = X[n_train:], y_arr[n_train:]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_list)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, feature_name=feature_list)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": float(settings.LGBM_LEARNING_RATE),
        "num_leaves": int(settings.LGBM_NUM_LEAVES),
        "min_data_in_leaf": int(settings.LGBM_MIN_DATA_IN_LEAF),
        "feature_fraction": float(settings.LGBM_FEATURE_FRACTION),
        "bagging_fraction": float(settings.LGBM_BAGGING_FRACTION),
        "bagging_freq": int(settings.LGBM_BAGGING_FREQ),
        "lambda_l2": float(settings.LGBM_L2_REG),
        "seed": int(settings.LGBM_SEED),
        "verbosity": -1,
    }

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=int(settings.LGBM_NUM_BOOST_ROUND),
        valid_sets=[dvalid],
        valid_names=["valid"],
    )

    p_valid = booster.predict(X_valid)
    # logloss
    p_clip = np.clip(p_valid, 1e-9, 1.0 - 1e-9)
    ll = float(-(y_valid * np.log(p_clip) + (1 - y_valid) * np.log(1 - p_clip)).mean())
    auc = _auc(y_valid, p_valid)

    model_str = booster.model_to_string(num_iteration=booster.best_iteration or booster.current_iteration())
    artifact_sha = sha256_hex(model_str.encode("utf-8"))

    metrics = {
        "n_samples": int(len(y_arr)),
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "valid_logloss": ll,
        "valid_auc": auc,
        "pos_rate": float(y_arr.mean()),
    }

    # deactivate previous active
    session.execute(
        update(models.ModelArtifact)
        .where(models.ModelArtifact.objective == obj, models.ModelArtifact.is_active == True)  # noqa: E712
        .values(is_active=False)
    )

    row = models.ModelArtifact(
        objective=obj,
        model_version=mv,
        feature_schema_version=str(settings.FEATURE_EXTRACTOR_VERSION or "v1"),
        is_active=True,
        metrics=metrics,
        feature_list=feature_list,
        artifact_text=model_str,
        artifact_sha256=artifact_sha,
        note=f"trained from {len(y_arr)} samples; lv={lv}",
    )
    session.add(row)
    session.flush()

    return TrainResult(
        objective=obj,
        model_version=mv,
        metrics=metrics,
        feature_list=feature_list,
        artifact_sha256=artifact_sha,
    )


def load_active_booster(session: Session, objective: str) -> Optional[lgb.Booster]:
    obj = objective.strip().upper()
    row = session.execute(
        select(models.ModelArtifact)
        .where(models.ModelArtifact.objective == obj, models.ModelArtifact.is_active == True)  # noqa: E712
        .order_by(models.ModelArtifact.trained_ts.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    try:
        return lgb.Booster(model_str=row.artifact_text)
    except Exception:
        return None


def predict_proba(
    booster: lgb.Booster,
    feature_list: list[str],
    features: dict[str, float],
) -> float:
    x = np.zeros((1, len(feature_list)), dtype=np.float32)
    for j, k in enumerate(feature_list):
        x[0, j] = float(features.get(k, 0.0))
    p = float(booster.predict(x)[0])
    return max(0.0, min(1.0, p))


def predict_contrib(
    booster: lgb.Booster,
    feature_list: list[str],
    features: dict[str, float],
) -> dict[str, float]:
    """Return per-feature contribution (SHAP-style) without the bias term."""
    x = np.zeros((1, len(feature_list)), dtype=np.float32)
    for j, k in enumerate(feature_list):
        x[0, j] = float(features.get(k, 0.0))
    contrib = booster.predict(x, pred_contrib=True)
    # shape: (1, n_features + 1), last is bias
    arr = contrib[0]
    out: dict[str, float] = {}
    for j, k in enumerate(feature_list):
        out[k] = float(arr[j])
    return out
