from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.database import models
from app.utils.crypto import sha256_hex
from app.utils.time import now_shanghai


# ---------------------------
# Feature engineering (P1)
# ---------------------------

_NUMERIC_KEYS = (
    "p_limit_up",
    "high_days_n",
    "turnover_rate",
    "order_amount",
    "sum_market_value",
    "is_again_limit",
    "change_rate",
)

# Categorical -> one hot (keep tiny on purpose)
_CAT_LIMITUP_TYPES = ("一字板", "换手板")


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


def build_features_from_snapshot(snapshot: dict[str, Any]) -> dict[str, float]:
    """Turn RAW_SNAPSHOT fields into a stable small feature vector.

    IMPORTANT:
    - Keep this deterministic and versioned via algo name in ModelArtifact.
    - Missing values are encoded as 0.0 with an additional *_missing flag where helpful.
    """
    f: dict[str, float] = {}

    # base numeric
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

    # limitup type one-hot
    lut = str(snapshot.get("limit_up_type") or "").strip()
    for t in _CAT_LIMITUP_TYPES:
        f[f"limit_up_type__{t}"] = 1.0 if lut == t else 0.0
    f["limit_up_type__OTHER"] = 1.0 if (lut and lut not in _CAT_LIMITUP_TYPES) else 0.0
    f["limit_up_type_missing"] = 1.0 if not lut else 0.0

    # a couple of gentle nonlinearities (cheap but helpful)
    # log scaling for large magnitude features
    f["log_order_amount"] = math.log1p(max(f["order_amount"], 0.0))
    f["log_sum_market_value"] = math.log1p(max(f["sum_market_value"], 0.0))

    # interaction-ish signals
    f["p_x_high_days"] = f["p_limit_up"] * f["high_days_n"]
    f["p_x_turnover"] = f["p_limit_up"] * f["turnover_rate"]

    return f


# ---------------------------
# Minimal logistic regression (no sklearn dependency)
# ---------------------------

@dataclass
class TrainedLogReg:
    feature_list: list[str]
    mean: dict[str, float]
    std: dict[str, float]
    weights: dict[str, float]
    bias: float
    temperature: float
    metrics: dict[str, Any]


def _sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logloss(p: float, y: int) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -math.log(p) if y == 1 else -math.log(1 - p)


def _auc(pairs: list[tuple[float, int]]) -> float | None:
    # Mann–Whitney U approximation
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return None
    # rank all
    pairs_sorted = sorted([(p, y) for p, y in pairs], key=lambda x: x[0])
    ranks = {}
    # average ranks for ties
    i = 0
    n = len(pairs_sorted)
    while i < n:
        j = i
        while j < n and pairs_sorted[j][0] == pairs_sorted[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_pos_ranks = 0.0
    for idx, (p, y) in enumerate(pairs_sorted):
        if y == 1:
            sum_pos_ranks += ranks[idx]
    n_pos = len(pos)
    n_neg = len(neg)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _standardize(rows: list[dict[str, float]], feature_list: list[str]) -> tuple[dict[str, float], dict[str, float], list[list[float]]]:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    X: list[list[float]] = []

    # compute mean
    for f in feature_list:
        s = 0.0
        for r in rows:
            s += float(r.get(f, 0.0))
        mean[f] = s / max(1, len(rows))

    # compute std
    for f in feature_list:
        ss = 0.0
        mu = mean[f]
        for r in rows:
            v = float(r.get(f, 0.0)) - mu
            ss += v * v
        var = ss / max(1, len(rows))
        std[f] = math.sqrt(var) if var > 1e-12 else 1.0

    # standardize
    for r in rows:
        X.append([(float(r.get(f, 0.0)) - mean[f]) / std[f] for f in feature_list])

    return mean, std, X


def _predict_proba_internal(x_std: list[float], w: list[float], b: float, temperature: float) -> float:
    z = b
    for xi, wi in zip(x_std, w):
        z += xi * wi
    z = z / max(1e-6, float(temperature))
    return _sigmoid(z)


def fit_logreg(
    rows: list[dict[str, float]],
    y: list[int],
    *,
    l2: float = 1.0,
    lr: float = 0.05,
    epochs: int = 400,
) -> TrainedLogReg:
    if len(rows) != len(y):
        raise ValueError("rows/y length mismatch")

    # choose feature_list deterministically
    feature_set: set[str] = set()
    for r in rows:
        feature_set.update(r.keys())
    feature_list = sorted(feature_set)

    mean, std, X = _standardize(rows, feature_list)

    # class weights for imbalance
    n = len(y)
    n_pos = sum(1 for v in y if v == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both positive and negative labels to train")
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)

    w = [0.0 for _ in feature_list]
    b = 0.0

    for _ep in range(int(epochs)):
        gw = [0.0 for _ in feature_list]
        gb = 0.0
        for xi, yi in zip(X, y):
            p = _sigmoid(sum(a * ww for a, ww in zip(xi, w)) + b)
            weight = w_pos if yi == 1 else w_neg
            diff = (p - yi) * weight
            for j in range(len(feature_list)):
                gw[j] += diff * xi[j]
            gb += diff

        # L2
        for j in range(len(feature_list)):
            gw[j] += l2 * w[j]

        # update
        for j in range(len(feature_list)):
            w[j] -= lr * gw[j] / max(1.0, n)
        b -= lr * gb / max(1.0, n)

    # temperature scaling (1D search)
    def nll(temp: float) -> float:
        s = 0.0
        for xi, yi in zip(X, y):
            p = _predict_proba_internal(xi, w, b, temp)
            s += _logloss(p, yi)
        return s / max(1, n)

    # grid search (robust)
    best_t = 1.0
    best = nll(best_t)
    for t in [0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0]:
        v = nll(t)
        if v < best:
            best = v
            best_t = t

    # metrics
    pairs = []
    brier = 0.0
    ll = 0.0
    for xi, yi in zip(X, y):
        p = _predict_proba_internal(xi, w, b, best_t)
        pairs.append((p, yi))
        brier += (p - yi) ** 2
        ll += _logloss(p, yi)
    auc = _auc(pairs)
    metrics = {
        "n_samples": n,
        "pos_rate": n_pos / max(1, n),
        "logloss": ll / max(1, n),
        "brier": brier / max(1, n),
        "auc": auc,
        "temperature": best_t,
        "l2": l2,
        "lr": lr,
        "epochs": epochs,
    }

    weights = {f: float(wi) for f, wi in zip(feature_list, w)}
    return TrainedLogReg(
        feature_list=feature_list,
        mean=mean,
        std=std,
        weights=weights,
        bias=float(b),
        temperature=float(best_t),
        metrics=metrics,
    )


def _extract_training_rows(s: Session, *, max_rows: int = 20000) -> tuple[list[dict[str, float]], list[int]]:
    """Build training set from:
    DecisionLabel (y) + DecisionEvidence(RAW_SNAPSHOT) (x).
    """
    # Latest label per decision_id wins (unique already by decision_id+label_day).
    # We train on hit_limitup as a first objective.
    q = (
        select(models.DecisionLabel, models.DecisionEvidence)
        .join(models.DecisionEvidence, models.DecisionEvidence.decision_id == models.DecisionLabel.decision_id)
        .where(models.DecisionEvidence.reason_code == "RAW_SNAPSHOT")
        .order_by(models.DecisionLabel.label_day.desc())
    )
    rows: list[dict[str, float]] = []
    ys: list[int] = []

    for dl, ev in s.execute(q).all():
        snap = dict(ev.evidence_fields or {})
        feats = build_features_from_snapshot(snap)
        rows.append(feats)
        ys.append(1 if bool(dl.hit_limitup) else 0)
        if len(rows) >= max_rows:
            break
    return rows, ys


def train_and_activate_model(s: Session, *, note: str = "") -> models.ModelArtifact:
    """Train a P1 baseline model and set it as active.

    This is intentionally simple and deterministic:
    - Train on all available labeled samples.
    - Store weights/scaler in DB (no external artifact deps).
    """
    rows, ys = _extract_training_rows(s)
    if len(rows) < 50:
        raise ValueError(f"not enough labeled samples to train (need >=50, got {len(rows)})")

    trained = fit_logreg(rows, ys)

    params = {
        "bias": trained.bias,
        "weights": trained.weights,
        "mean": trained.mean,
        "std": trained.std,
        "temperature": trained.temperature,
        "algo": "LOGREG_V1",
        "feature_list": trained.feature_list,
    }
    model_id = sha256_hex(json_dumps_sorted(params).encode("utf-8"))[:64]

    # deactivate older
    s.execute(update(models.ModelArtifact).values(is_active=False))

    ma = models.ModelArtifact(
        model_id=model_id,
        algo="LOGREG_V1",
        feature_list=trained.feature_list,
        params=params,
        metrics=trained.metrics,
        is_active=True,
        note=note or "",
        created_ts=now_shanghai(),
    )
    s.add(ma)
    s.flush()
    return ma


def json_dumps_sorted(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def load_active_model(s: Session) -> models.ModelArtifact | None:
    return (
        s.execute(select(models.ModelArtifact).where(models.ModelArtifact.is_active == True).order_by(models.ModelArtifact.created_ts.desc()))
        .scalars()
        .first()
    )


def predict_proba_from_model(model: models.ModelArtifact, snapshot: dict[str, Any]) -> tuple[float, list[tuple[str, float]]]:
    """Return (p, contributions).

    contributions are (feature, contribution) in logit space for explainability.
    """
    p = 0.0
    contribs: list[tuple[str, float]] = []

    try:
        feats = build_features_from_snapshot(snapshot)
        params = dict(model.params or {})
        feature_list = list(params.get("feature_list") or model.feature_list or [])
        weights: dict[str, float] = dict(params.get("weights") or {})
        mean: dict[str, float] = dict(params.get("mean") or {})
        std: dict[str, float] = dict(params.get("std") or {})
        bias = float(params.get("bias") or 0.0)
        temperature = float(params.get("temperature") or 1.0)

        z = bias
        for f in feature_list:
            x = float(feats.get(f, 0.0))
            mu = float(mean.get(f, 0.0))
            sd = float(std.get(f, 1.0)) or 1.0
            x_std = (x - mu) / sd
            w = float(weights.get(f, 0.0))
            c = x_std * w
            z += c
            if abs(c) > 1e-8:
                contribs.append((f, float(c)))
        z_t = z / max(1e-6, temperature)
        p = _sigmoid(z_t)

        # sort by absolute contribution
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        contribs = contribs[:6]
        return float(p), contribs
    except Exception:
        return 0.0, []
