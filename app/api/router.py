from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Body
from sqlalchemy import select, func, update

from app.database.engine import SessionLocal
from app.database.repo import Repo
from app.database import models
from app.utils.time import now_shanghai_str, now_shanghai, to_shanghai, trading_day_str
from app.utils.crypto import sha256_hex
from app.utils.symbols import normalize_symbol
from app.config import settings
from app.core.labeling_planner import build_plan
from app.adapters.pool_fetcher import fetch_limitup_pool
from app.core.recommender_v2 import generate_for_batch_v2, persist_decisions_v2
from app.core.collector_pipeline import run_collectors_for_committed_batch
from app.core.cutoff_tasks import run_eod_cutoff_1530
from app.core.model_training import (
    ensure_labels_for_candidates,
    train_objective_lightgbm,
    OBJ_TP5_3D,
    OBJ_TP8_3D,
)

from decimal import Decimal
from datetime import date, time


router = APIRouter()


def _parse_ui_day(day: str | None) -> tuple[str | None, datetime | None, datetime | None]:
    """Accepts YYYY-MM-DD or YYYYMMDD. Returns (trading_day_YYYYMMDD, start_dt, end_dt) in Asia/Shanghai."""
    if not day:
        return None, None, None
    d = day.strip()
    if not d:
        return None, None, None
    if len(d) == 8 and d.isdigit():
        td = d
        dt = datetime.strptime(td, "%Y%m%d")
    else:
        # allow YYYY-MM-DD
        dt = datetime.strptime(d, "%Y-%m-%d")
        td = dt.strftime("%Y%m%d")
    start = to_shanghai(dt).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return td, start, end


@router.get("/health")
def health() -> dict:
    return {"ok": True, "time": now_shanghai_str()}


@router.get("/status")
def status() -> dict:
    with SessionLocal() as s:
        repo = Repo(s)
        st = repo.system_status.get_for_update()
        s.commit()
        return {
            "time": now_shanghai_str(),
            "panic_halt": bool(st.panic_halt),
            "guard_level": int(st.guard_level),
            "veto": bool(st.veto),
            "veto_code": st.veto_code,
            "last_self_check_report_hash": st.last_self_check_report_hash,
            "last_self_check_time": st.last_self_check_time.isoformat() if st.last_self_check_time else None,
        }


@router.post("/admin/self_check")
def run_self_check() -> dict:
    """
    Minimal self-check report, required by trade gate in this skeleton.
    """
    with SessionLocal() as s:
        repo = Repo(s)
        _st = repo.system_status.get_for_update()

        report = {
            "time": now_shanghai_str(),
            "versions": {
                "RuleSetVersionHash": settings.RULESET_VERSION_HASH,
                "StrategyContractHash": settings.STRATEGY_CONTRACT_HASH,
                "ModelSnapshotUUID": settings.MODEL_SNAPSHOT_UUID,
                "CostModelVersion": settings.COST_MODEL_VERSION,
                "CanonicalizationVersion": settings.CANONICALIZATION_VERSION,
                "FeatureExtractorVersion": settings.FEATURE_EXTRACTOR_VERSION,
            },
            "note": "minimal self-check: schema+versions+connectivity assumed OK",
        }
        report_hash = sha256_hex(str(report).encode("utf-8"))
        repo.system_status.set_self_check(report_hash)
        repo.system_events.write_event(
            event_type="SELF_CHECK_REPORT",
            correlation_id=None,
            severity="INFO",
            payload={"report": report, "report_hash": report_hash},
        )
        s.commit()
        return {"ok": True, "report_hash": report_hash, "report": report}


# ---------------------------
# Admin: Calendar / Daily OHLCV baseline data
# ---------------------------


def _parse_day_to_date(d: str) -> date:
    s = str(d or "").strip()
    if not s:
        raise ValueError("day is required")
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s, "%Y-%m-%d").date()


@router.post("/admin/calendar/upsert")
def upsert_trading_calendar(payload: list[dict] = Body(...)) -> dict:
    """Upsert trading calendar days.

    Body: [{"day":"2026-01-16"|"20260116", "is_open":true, "note":"..."}, ...]
    """
    if not isinstance(payload, list) or not payload:
        raise HTTPException(status_code=400, detail="payload must be a non-empty list")

    ins = 0
    upd = 0
    with SessionLocal() as s:
        for item in payload:
            try:
                day = _parse_day_to_date(item.get("day"))
                is_open = bool(item.get("is_open", True))
                note = str(item.get("note") or "") or None
            except Exception:
                continue

            row = s.get(models.TradingCalendarDay, day)
            if row is None:
                s.add(models.TradingCalendarDay(day=day, is_open=is_open, note=note))
                ins += 1
            else:
                row.is_open = is_open
                row.note = note
                upd += 1
        s.commit()

    return {"ok": True, "inserted": ins, "updated": upd}


@router.post("/admin/daily_ohlcv/upsert")
def upsert_daily_ohlcv(payload: list[dict] = Body(...)) -> dict:
    """Upsert daily OHLCV facts.

    Body: [{
      "instrument_id":"SZ000001",
      "trading_day":"2026-01-15"|"20260115",
      "open": 10.1, "high": 10.5, "low": 9.9, "close": 10.2,
      "volume": 123456.0, "amount": 987654321.0,
      "source":"THS", "raw_hash":"..."}
    ]
    """
    if not isinstance(payload, list) or not payload:
        raise HTTPException(status_code=400, detail="payload must be a non-empty list")

    ins = 0
    upd = 0
    skipped = 0
    with SessionLocal() as s:
        for item in payload:
            try:
                inst = normalize_symbol(item.get("instrument_id") or item.get("symbol"))
                td = _parse_day_to_date(item.get("trading_day") or item.get("day"))
                o = Decimal(str(item.get("open")))
                h = Decimal(str(item.get("high")))
                l = Decimal(str(item.get("low")))
                c = Decimal(str(item.get("close")))
                vol = item.get("volume")
                amt = item.get("amount")
                source = str(item.get("source") or "UNKNOWN")
                raw_hash = str(item.get("raw_hash") or "") or None
            except Exception:
                skipped += 1
                continue

            row = (
                s.execute(
                    select(models.FactDailyOHLCV).where(
                        models.FactDailyOHLCV.instrument_id == inst,
                        models.FactDailyOHLCV.trading_day == td,
                    )
                )
                .scalars()
                .one_or_none()
            )
            if row is None:
                s.add(
                    models.FactDailyOHLCV(
                        instrument_id=inst,
                        trading_day=td,
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=Decimal(str(vol)) if vol is not None else None,
                        amount=Decimal(str(amt)) if amt is not None else None,
                        source=source,
                        raw_hash=raw_hash,
                    )
                )
                ins += 1
            else:
                row.open = o
                row.high = h
                row.low = l
                row.close = c
                row.volume = Decimal(str(vol)) if vol is not None else None
                row.amount = Decimal(str(amt)) if amt is not None else None
                row.source = source
                row.raw_hash = raw_hash
                upd += 1

        s.commit()

    return {"ok": True, "inserted": ins, "updated": upd, "skipped": skipped}


# ---------------------------
# Admin: Labels & Model Training (LightGBM)
# ---------------------------


@router.post("/admin/ml/labels3d")
def build_labels_3d(trading_day: str = Body(..., embed=True), label_version: str | None = Body(None, embed=True)) -> dict:
    """Generate 3D labels for the candidate pool on a given trading_day (T).

    It will create rows in `model_training_label_3d` for symbols that have
    the required Daily OHLCV facts for T, T+1..T+3.

    Body:
      {"trading_day":"20260115"|"2026-01-15", "label_version":"v1"}
    """
    try:
        td = trading_day.strip()
        if len(td) == 10 and "-" in td:
            td = datetime.strptime(td, "%Y-%m-%d").strftime("%Y%m%d")
        if not (len(td) == 8 and td.isdigit()):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="invalid trading_day")

    with SessionLocal() as s:
        cnt = ensure_labels_for_candidates(s, trading_day=td, label_version=label_version)
        s.commit()
        return {"ok": True, "trading_day": td, "labels_upserted": cnt, "label_version": (label_version or settings.LABEL3D_VERSION)}


@router.post("/admin/ml/train")
def train_models(label_version: str | None = Body(None, embed=True), max_rows: int = Body(5000, embed=True)) -> dict:
    """Train and activate LightGBM models for TP5_3D and TP8_3D.

    Body: {"label_version":"v1", "max_rows":5000}
    """
    with SessionLocal() as s:
        r5 = train_objective_lightgbm(s, OBJ_TP5_3D, label_version=label_version, max_rows=int(max_rows))
        r8 = train_objective_lightgbm(s, OBJ_TP8_3D, label_version=label_version, max_rows=int(max_rows))
        s.commit()
        return {
            "ok": True,
            "trained": [
                {"objective": r5.objective, "model_version": r5.model_version, "metrics": r5.metrics, "artifact_sha256": r5.artifact_sha256},
                {"objective": r8.objective, "model_version": r8.model_version, "metrics": r8.metrics, "artifact_sha256": r8.artifact_sha256},
            ],
        }

@router.post("/admin/ml/pipeline")
def run_pipeline(payload: dict = Body(...)) -> dict:
    """One-key pipeline: backfill OHLCV -> build labels -> train & activate.

    Body example:
      {
        "start_day": "2026-01-01",
        "end_day": "2026-01-15",
        "label_version": "v1",
        "do_backfill": true,
        "do_labels": true,
        "do_train": true,
        "max_symbols_per_day": 5000,
        "max_rows": 5000
      }

    Notes:
    - Backfill uses DATA_PROVIDER (IFIND_HTTP / THS_DATAAPI / CUSTOM_HTTP / MOCK).
    - Labels follow entry=Open(T+1) and max High(T+1..T+3).
    """
    from app.core.ml_pipeline import run_ml_pipeline

    start_day = str(payload.get("start_day") or "").strip()
    end_day = str(payload.get("end_day") or "").strip()
    if not start_day or not end_day:
        raise HTTPException(status_code=400, detail="start_day/end_day required")

    label_version = payload.get("label_version")
    do_backfill = bool(payload.get("do_backfill", True))
    do_labels = bool(payload.get("do_labels", True))
    do_train = bool(payload.get("do_train", True))
    max_symbols_per_day = int(payload.get("max_symbols_per_day", 5000))
    max_rows = int(payload.get("max_rows", 5000))

    with SessionLocal() as s:
        res = run_ml_pipeline(
            s,
            start_day=start_day,
            end_day=end_day,
            label_version=label_version,
            do_backfill=do_backfill,
            do_labels=do_labels,
            do_train=do_train,
            max_symbols_per_day=max_symbols_per_day,
            max_rows=max_rows,
        )
        s.commit()
        return {
            "ok": True,
            "trading_days": res.trading_days,
            "backfill": res.backfill,
            "labels": res.labels,
            "train": res.train,
            "activated": res.activated,
            "started_ts": res.started_ts,
            "finished_ts": res.finished_ts,
        }



@router.get("/admin/ml/models")
def list_models(limit: int = 50) -> list[dict]:
    """List recent model artifacts (including active flag)."""
    with SessionLocal() as s:
        rows = (
            s.execute(
                select(models.ModelArtifact)
                .order_by(models.ModelArtifact.trained_ts.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        out: list[dict] = []
        for r in rows:
            out.append(
                {
                    "objective": r.objective,
                    "model_version": r.model_version,
                    "feature_schema_version": r.feature_schema_version,
                    "is_active": bool(r.is_active),
                    "metrics": r.metrics or {},
                    "feature_list": list(r.feature_list or []),
                    "artifact_sha256": r.artifact_sha256,
                    "trained_ts": r.trained_ts.isoformat() if r.trained_ts else None,
                    "note": r.note,
                }
            )
        return out


@router.get("/admin/pool_rules")
def get_pool_rules() -> dict:
    """Inspect current effective pool filter rules.

    Note: switching the rules source (ENV/DB/VERSIONED) is controlled by env var POOL_RULES_SOURCE.
    """
    from app.core.orchestrator import get_pool_filter_rules

    with SessionLocal() as s:
        prefixes, exchanges, meta = get_pool_filter_rules(s)
        return {
            "pool_rules_source": (settings.POOL_RULES_SOURCE or "ENV").upper(),
            "effective": meta,
            "allowed_prefixes": sorted(list(prefixes)),
            "allowed_exchanges": sorted(list(exchanges)),
            "time": now_shanghai_str(),
        }


@router.put("/admin/pool_rules")
def set_pool_rules(payload: dict = Body(...)) -> dict:
    """Update DB-backed pool rules (方案2).

    Takes either CSV strings or lists:
      {"allowed_prefixes": "0,6", "allowed_exchanges": ["SZ","SH"]}
    """
    def _to_list(v) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip().upper() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [x.strip().upper() for x in v.split(",") if x.strip()]
        return [str(v).strip().upper()] if str(v).strip() else []

    prefixes = _to_list(payload.get("allowed_prefixes"))
    exchanges = _to_list(payload.get("allowed_exchanges"))
    if not prefixes and not exchanges:
        raise HTTPException(status_code=400, detail="allowed_prefixes/allowed_exchanges is required")

    with SessionLocal() as s:
        repo = Repo(s)
        if prefixes:
            repo.system_settings.set("pool.allowed_prefixes", prefixes)
        if exchanges:
            repo.system_settings.set("pool.allowed_exchanges", exchanges)
        s.commit()

    return {
        "ok": True,
        "saved": {
            "pool.allowed_prefixes": prefixes or None,
            "pool.allowed_exchanges": exchanges or None,
        },
        "note": "Effective only when POOL_RULES_SOURCE is DB or VERSIONED(fallback).",
    }


@router.post("/admin/pool_rulesets")
def create_pool_ruleset(payload: dict = Body(...)) -> dict:
    """Create a versioned pool filter rule set (方案3).

    Payload:
      {
        "allowed_prefixes": ["0","6"],
        "allowed_exchanges": "SZ,SH",
        "effective_ts": "2026-01-15 16:00:00",
        "note": "expand to SH"
      }
    """
    def _to_list(v) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip().upper() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [x.strip().upper() for x in v.split(",") if x.strip()]
        return [str(v).strip().upper()] if str(v).strip() else []

    prefixes = _to_list(payload.get("allowed_prefixes"))
    exchanges = _to_list(payload.get("allowed_exchanges"))
    if not prefixes or not exchanges:
        raise HTTPException(status_code=400, detail="allowed_prefixes and allowed_exchanges are required")

    eff = payload.get("effective_ts")
    if eff:
        try:
            # accept YYYY-MM-DD HH:MM:SS or ISO
            if isinstance(eff, str) and "T" in eff:
                dt = datetime.fromisoformat(eff)
            elif isinstance(eff, str) and len(eff.strip()) <= 10:
                dt = datetime.strptime(eff.strip(), "%Y-%m-%d")
            else:
                dt = datetime.strptime(str(eff).strip(), "%Y-%m-%d %H:%M:%S")
        except Exception:
            raise HTTPException(status_code=400, detail="effective_ts invalid; use ISO or 'YYYY-MM-DD HH:MM:SS'")
        dt = to_shanghai(dt)
    else:
        dt = now_shanghai()

    note = str(payload.get("note") or "")

    with SessionLocal() as s:
        repo = Repo(s)
        row = repo.pool_filter_rules.create(prefixes, exchanges, dt, note=note)
        s.commit()

        return {
            "ok": True,
            "rule_set_id": row.rule_set_id,
            "allowed_prefixes": list(row.allowed_prefixes or []),
            "allowed_exchanges": list(row.allowed_exchanges or []),
            "effective_ts": row.effective_ts.isoformat(),
            "note": row.note,
            "hint": "Set POOL_RULES_SOURCE=VERSIONED to make this take effect automatically.",
        }


@router.get("/admin/pool_rulesets")
def list_pool_rulesets(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        repo = Repo(s)
        rows = repo.pool_filter_rules.list_recent(limit=limit)
        return [
            {
                "rule_set_id": r.rule_set_id,
                "allowed_prefixes": list(r.allowed_prefixes or []),
                "allowed_exchanges": list(r.allowed_exchanges or []),
                "effective_ts": r.effective_ts.isoformat(),
                "note": r.note,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ---------------------------
# Candidate pool (Canonical Schema v1)
# ---------------------------


@router.post("/pool/fetch_now")
def pool_fetch_now() -> dict:
    """Manual trigger: fetch external candidate pool immediately (for ops/testing)."""
    res = fetch_limitup_pool()

    # reuse the same filtering rule as orchestrator
    from app.core.orchestrator import _filter_and_normalize_items, get_pool_filter_rules
    td = trading_day_str(now_shanghai())
    with SessionLocal() as s:
        allowed_prefixes, allowed_exchanges, rules = get_pool_filter_rules(s)
    filtered = _filter_and_normalize_items(res.items, allowed_prefixes, allowed_exchanges)
    mode = str(getattr(settings, "POOL_FETCHER_MODE", "CUSTOM") or "CUSTOM").strip().upper()
    src = settings.POOL_FETCH_URL if mode != "THS_10JQKA" else "THS_10JQKA"
    batch_id = sha256_hex(f"{td}|{src}|{res.raw_hash}".encode("utf-8"))[:32]

    with SessionLocal() as s:
        # idempotent: if exists, return existing summary
        b = s.get(models.LimitupPoolBatch, batch_id)
        if b is None:
            s.add(
                models.LimitupPoolBatch(
                    batch_id=batch_id,
                    trading_day=td,
                    fetch_ts=now_shanghai(),
                    source=("THS_10JQKA" if mode == "THS_10JQKA" else "EXTERNAL"),
                    status="EDITING",
                    filter_rules=rules,
                    raw_hash=res.raw_hash,
                )
            )
            s.flush()
            for it in filtered:
                s.add(
                    models.LimitupCandidate(
                        batch_id=batch_id,
                        symbol=normalize_symbol(str(it.get("symbol") or it.get("code") or "")),
                        name=str(it.get("name") or ""),
                        p_limit_up=None,
                        p_source="UI",
                        edited_ts=None,
                        candidate_status="PENDING_EDIT",
                        raw_json=it,
                    )
                )
            s.commit()

        cnt = (
            s.execute(select(func.count(models.LimitupCandidate.id)).where(models.LimitupCandidate.batch_id == batch_id)).scalar_one()
        )
        return {
            "batch_id": batch_id,
            "trading_day": datetime.strptime(td, "%Y%m%d").strftime("%Y-%m-%d"),
            "status": (b.status if b else "EDITING"),
            "raw_hash": res.raw_hash,
            "filtered_count": int(cnt or 0),
            "filter_rules": rules,
        }


@router.get("/pool/batches")
def list_pool_batches(limit: int = 30, day: str | None = None) -> list[dict]:
    """List pool batches."""
    td, _, _ = _parse_ui_day(day)
    with SessionLocal() as s:
        q = select(models.LimitupPoolBatch).order_by(models.LimitupPoolBatch.fetch_ts.desc())
        if td:
            q = q.where(models.LimitupPoolBatch.trading_day == td)
        rows = s.execute(q.limit(limit)).scalars().all()
        return [
            {
                "batch_id": r.batch_id,
                "trading_day": datetime.strptime(r.trading_day, "%Y%m%d").strftime("%Y-%m-%d"),
                "fetch_ts": r.fetch_ts.isoformat(),
                "source": r.source,
                "status": r.status,
                "filter_rules": r.filter_rules,
                "raw_hash": r.raw_hash,
            }
            for r in rows
        ]


@router.get("/pool/batches/{batch_id}/candidates")
def list_pool_candidates(
    batch_id: str,
    limit: int = 500,
    status: str | None = None,
    include_all: bool = False,
) -> list[dict]:
    with SessionLocal() as s:
        # Default behavior is optimized for the editor UI:
        # - hide already-edited candidates (READY/DROPPED) unless explicitly requested.
        if not include_all and (status is None):
            status = "PENDING_EDIT"

        q = select(models.LimitupCandidate).where(models.LimitupCandidate.batch_id == batch_id)
        if status:
            q = q.where(models.LimitupCandidate.candidate_status == str(status).strip())
        rows = (
            s.execute(q.order_by(models.LimitupCandidate.p_limit_up.desc().nullslast()).limit(limit))
            .scalars()
            .all()
        )
        return [
            {
                "batch_id": r.batch_id,
                "symbol": normalize_symbol(r.symbol),
                "name": r.name,
                "p_limit_up": (float(r.p_limit_up) if r.p_limit_up is not None else None),
                "p_source": r.p_source,
                "edited_ts": r.edited_ts.isoformat() if r.edited_ts else None,
                "candidate_status": r.candidate_status,
                "raw_json": r.raw_json,
            }
            for r in rows
        ]


@router.patch("/pool/batches/{batch_id}/candidates/{symbol}")
def update_pool_candidate(batch_id: str, symbol: str, payload: dict = Body(...)) -> dict:
    """Update operator-edited fields (p_limit_up) for a candidate."""
    p = payload.get("p_limit_up")
    if p is None:
        raise HTTPException(status_code=400, detail="p_limit_up is required")
    try:
        pf = float(p)
    except Exception:
        raise HTTPException(status_code=400, detail="p_limit_up must be a number")
    if not (0.0 <= pf <= 1.0):
        raise HTTPException(status_code=400, detail="p_limit_up must be in [0,1]")

    sym = normalize_symbol(symbol)
    with SessionLocal() as s:
        row = (
            s.execute(
                select(models.LimitupCandidate)
                .where(models.LimitupCandidate.batch_id == batch_id)
                .where(models.LimitupCandidate.symbol == sym)
            )
            .scalars()
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="candidate_not_found")

        # mark batch as EDITING on first operator edit
        b = s.get(models.LimitupPoolBatch, batch_id)
        if b is not None and b.status == "FETCHED":
            b.status = "EDITING"

        row.p_limit_up = pf
        row.p_source = str(payload.get("p_source") or "UI")
        row.edited_ts = now_shanghai()
        # once edited, mark READY unless explicitly dropped
        if str(payload.get("candidate_status") or "").strip():
            row.candidate_status = str(payload.get("candidate_status")).strip()
        elif row.candidate_status != "DROPPED":
            row.candidate_status = "READY"

        s.commit()
        return {
            "batch_id": batch_id,
            "symbol": sym,
            "p_limit_up": float(row.p_limit_up),
            "candidate_status": row.candidate_status,
            "edited_ts": row.edited_ts.isoformat() if row.edited_ts else None,
        }


@router.post("/pool/batches/{batch_id}/commit")
def commit_pool_batch(batch_id: str) -> dict:
    """Commit a batch and immediately generate recommendations (P0 v1)."""
    with SessionLocal() as s:
        b = s.get(models.LimitupPoolBatch, batch_id)
        if b is None:
            raise HTTPException(status_code=404, detail="batch_not_found")
        if b.status == "CANCELLED":
            raise HTTPException(status_code=400, detail="batch_cancelled")

        # Require at least one READY candidate with p_limit_up filled
        ready_cnt = (
            s.execute(
                select(func.count())
                .select_from(models.LimitupCandidate)
                .where(models.LimitupCandidate.batch_id == batch_id)
                .where(models.LimitupCandidate.candidate_status == "READY")
                .where(models.LimitupCandidate.p_limit_up.is_not(None))
            )
            .scalar_one()
        )
        if int(ready_cnt or 0) <= 0:
            raise HTTPException(status_code=400, detail="no_ready_candidates")

        # Drop any remaining PENDING_EDIT candidates to keep the batch auditable
        s.execute(
            update(models.LimitupCandidate)
            .where(models.LimitupCandidate.batch_id == batch_id)
            .where(models.LimitupCandidate.candidate_status == "PENDING_EDIT")
            .values(candidate_status="DROPPED")
        )
        # mark batch committed
        b.status = "COMMITTED"
        s.flush()

        # Run offline collectors (history/theme) immediately for this committed batch.
        run_collectors_for_committed_batch(s, b)

        # generate recommendations now
        items = generate_for_batch_v2(s, b)
        persist_decisions_v2(s, b, items)
        s.commit()

        return {
            "batch_id": b.batch_id,
            "trading_day": datetime.strptime(b.trading_day, "%Y%m%d").strftime("%Y-%m-%d"),
            "status": b.status,
            "recommendation_count": len(items),
        }


@router.get("/recommendations")
def list_recommendations(
    limit: int = 50,
    day: str | None = None,
    trading_day: str | None = None,
) -> list[dict]:
    """User-facing recommendations (TopN by score).

    Query priority:
      1) trading_day=YYYY-MM-DD|YYYYMMDD  -> filter by ModelDecision.trading_day (pool/source day, i.e. T)
      2) day=YYYY-MM-DD|YYYYMMDD          -> filter by ModelDecision.decision_day (decision day, i.e. T+1)
      3) no params                         -> latest available decision_day in DB (max), not by time arithmetic
    """
    with SessionLocal() as s:
        # 1) trading_day filter (pool day T)
        if trading_day:
            td, _, _ = _parse_ui_day(trading_day)
            if not td:
                raise HTTPException(status_code=400, detail="Invalid trading_day. Use YYYY-MM-DD or YYYYMMDD.")

            decs = (
                s.execute(
                    select(models.ModelDecision)
                    .where(models.ModelDecision.trading_day == td)
                    .order_by(models.ModelDecision.score.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        else:
            # 2) decision_day filter
            if day:
                dd, _, _ = _parse_ui_day(day)
                if not dd:
                    raise HTTPException(status_code=400, detail="Invalid day. Use YYYY-MM-DD or YYYYMMDD.")
                decision_day = dd
            else:
                # 3) default: latest available decision_day in DB
                decision_day = s.execute(select(func.max(models.ModelDecision.decision_day))).scalar_one()
                if not decision_day:
                    return []

            decs = (
                s.execute(
                    select(models.ModelDecision)
                    .where(models.ModelDecision.decision_day == decision_day)
                    .order_by(models.ModelDecision.score.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )

        if not decs:
            return []

        ids = [d.decision_id for d in decs]
        ev_rows = (
            s.execute(select(models.DecisionEvidence).where(models.DecisionEvidence.decision_id.in_(ids)))
            .scalars()
            .all()
        )
        ev_by_id: dict[str, list[models.DecisionEvidence]] = {}
        for ev in ev_rows:
            ev_by_id.setdefault(ev.decision_id, []).append(ev)

        out: list[dict] = []
        for d in decs:
            evs = ev_by_id.get(d.decision_id, [])
            out.append(
                {
                    "decision_id": d.decision_id,
                    "trading_day": datetime.strptime(d.trading_day, "%Y%m%d").strftime("%Y-%m-%d"),
                    "decision_day": datetime.strptime(d.decision_day, "%Y%m%d").strftime("%Y-%m-%d"),
                    "symbol": normalize_symbol(d.symbol),
                    "action": d.action,
                    "score": float(d.score),
                    "confidence": float(d.confidence),
                    "evidence": [
                        {
                            "reason_code": e.reason_code,
                            "reason_text": e.reason_text,
                            "evidence_fields": e.evidence_fields,
                            "evidence_refs": e.evidence_refs,
                        }
                        for e in evs
                    ],
                    "created_ts": d.created_ts.isoformat(),
                }
            )
        return out


@router.get("/decisions")
def list_decisions(limit: int = 50, day: str | None = None) -> list[dict]:
    td, start, end = _parse_ui_day(day)
    with SessionLocal() as s:
        q = select(models.DecisionBundle).order_by(models.DecisionBundle.created_at.desc())
        if start and end:
            q = q.where(models.DecisionBundle.created_at >= start, models.DecisionBundle.created_at < end)
        rows = s.execute(q.limit(limit)).scalars().all()
        return [
            {
                "decision_id": r.decision_id,
                "cid": r.cid,
                "account_id": r.account_id,
                "symbol": normalize_symbol(r.symbol),
                "decision": r.decision,
                "confidence": float((r.params or {}).get("confidence", 0.0)) if isinstance(r.params, dict) else None,
                "reason_code": r.reason_code,
                "params": r.params,
                "request_ids": r.request_ids,
                "model_hash": r.model_hash,
                "feature_hash": r.feature_hash,
                "guard_status": r.guard_status,
                "data_quality": r.data_quality,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


@router.get("/data_requests")
def list_data_requests(status: str | None = None, limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        q = select(models.DataRequest).order_by(models.DataRequest.created_at.desc())
        if status:
            q = q.where(models.DataRequest.status == status)
        rows = s.execute(q.limit(limit)).scalars().all()
        return [
            {
                "request_id": r.request_id,
                "dedupe_key": r.dedupe_key,
                "correlation_id": r.correlation_id,
                "account_id": r.account_id,
                "symbol": normalize_symbol(r.symbol),
                "purpose": r.purpose,
                "provider": r.provider,
                "endpoint": r.endpoint,
                "status": r.status,
                "attempts": r.attempts,
                "created_at": r.created_at.isoformat(),
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
                "deadline_at": r.deadline_at.isoformat() if r.deadline_at else None,
                "response_id": r.response_id,
                "last_error": r.last_error,
            }
            for r in rows
        ]


@router.get("/data_responses/{response_id}")
def get_data_response(response_id: str) -> dict:
    with SessionLocal() as s:
        r = s.get(models.DataResponse, response_id)
        if r is None:
            raise HTTPException(status_code=404, detail="not_found")
        return {
            "response_id": r.response_id,
            "request_id": r.request_id,
            "provider": r.provider,
            "endpoint": r.endpoint,
            "http_status": r.http_status,
            "errorcode": r.errorcode,
            "errmsg": r.errmsg,
            "quota_context": r.quota_context,
            "payload_sha256": r.payload_sha256,
            "received_at": r.received_at.isoformat(),
            "raw": r.raw,
        }


@router.get("/ui/signal_inputs")
def ui_signal_inputs(day: str) -> list[dict]:
    """List daily '待打标' candidates (preferred), fallback to internal Signals if none exist."""
    td, start, end = _parse_ui_day(day)
    if not td:
        raise HTTPException(status_code=400, detail="day is required (YYYY-MM-DD or YYYYMMDD)")

    with SessionLocal() as s:
        repo = Repo(s)

        # Preferred path: operator-supplied candidates (front-end填报/上传)
        cands = repo.labeling_candidates.list_by_day(td)
        if cands:
            trading_day_fmt = datetime.strptime(td, "%Y%m%d").strftime("%Y-%m-%d")
            target_td = cands[0].target_day or (datetime.strptime(td, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            target_day_fmt = datetime.strptime(target_td, "%Y%m%d").strftime("%Y-%m-%d")

            out: list[dict] = []
            for i, r in enumerate(cands, start=1):
                out.append(
                    {
                        "id": r.candidate_id,
                        "trading_day": trading_day_fmt,
                        "target_day": target_day_fmt,
                        "symbol": normalize_symbol(r.symbol),
                        "name": r.name,
                        "input_ts": r.updated_at.isoformat(),
                        "p_limit_up": float(r.p_limit_up),
                        "rank": i,
                        "source": r.source,
                        "extra": r.extra,
                    }
                )
            return out

        # Fallback: internal Signals pool (legacy behavior / dev mode)
        sigs = (
            s.execute(
                select(models.Signal)
                .where(models.Signal.trading_day == td)
                .order_by(models.Signal.confidence.desc())
            )
            .scalars()
            .all()
        )

        symbols = [x.symbol for x in sigs]
        latest_dec_by_sym: dict[str, models.DecisionBundle] = {}
        if symbols and start and end:
            decs = (
                s.execute(
                    select(models.DecisionBundle)
                    .where(models.DecisionBundle.created_at >= start, models.DecisionBundle.created_at < end)
                    .where(models.DecisionBundle.symbol.in_(symbols))
                    .order_by(models.DecisionBundle.created_at.desc())
                )
                .scalars()
                .all()
            )
            for drow in decs:
                if drow.symbol not in latest_dec_by_sym:
                    latest_dec_by_sym[drow.symbol] = drow

        target_day = (datetime.strptime(td, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        trading_day_fmt = datetime.strptime(td, "%Y%m%d").strftime("%Y-%m-%d")

        out: list[dict] = []
        for i, r in enumerate(sigs, start=1):
            dec = latest_dec_by_sym.get(r.symbol)
            params = dec.params if (dec and isinstance(dec.params, dict)) else {}
            out.append(
                {
                    "id": r.cid,
                    "trading_day": trading_day_fmt,
                    "target_day": target_day,
                    "symbol": normalize_symbol(r.symbol),
                    "strategy_id": r.strategy_id,
                    "input_ts": r.signal_ts.isoformat(),
                    # Proxy: p_limit_up currently equals model confidence (0..1)
                    "p_limit_up": float(r.confidence),
                    "rank": i,
                    "reason_code": (dec.reason_code if dec else ""),
                    "top_features": params.get("top_features"),
                    "features_snapshot": params.get("features_snapshot") or params.get("evidence") or params,
                }
            )
        return out


@router.post("/ui/signal_inputs")
def upsert_ui_signal_inputs(payload: dict = Body(...)) -> dict:
    """Upsert daily candidates for labeling (待打标).

    Expected payload:
    {
      "day": "YYYY-MM-DD" | "YYYYMMDD",
      "items": [{"symbol": "...", "p_limit_up": 0.23, "name": "...", ...}, ...]
    }
    """
    day = str(payload.get("day") or payload.get("trading_day") or "").strip()
    if not day:
        day = now_shanghai().strftime("%Y-%m-%d")

    td, _, _ = _parse_ui_day(day)
    if not td:
        raise HTTPException(status_code=400, detail="Invalid day. Use YYYY-MM-DD or YYYYMMDD.")

    items = payload.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="items must be a list")

    target_td = (datetime.strptime(td, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    with SessionLocal() as s:
        repo = Repo(s)
        res = repo.labeling_candidates.upsert_batch(trading_day=td, target_day=target_td, items=items, source="UI")

        # Update watchlist (symbols may appear repeatedly across days).
        # When LABELING_AUTO_FETCH_ENABLED is on, enqueue a model-driven (planner) data fetch plan.
        planned = 0
        for it in items:
            symbol = normalize_symbol(str((it or {}).get("symbol") or "").strip())
            if not symbol:
                continue

            wl = repo.watchlist.upsert_hit(symbol=symbol, trading_day=td)

            if settings.LABELING_AUTO_FETCH_ENABLED and wl.active:
                plan = build_plan(symbol=symbol, hit_count=int(wl.hit_count), planner_state=dict(wl.planner_state or {}))
                # Persist planner state back
                wl.planner_state = plan.planner_state

                for pr in plan.requests:
                    # HARD GUARANTEE:
                    # planned request must carry symbol; do not let repo enqueue_planned see symbol=None/""
                    try:
                        setattr(pr, "symbol", symbol)
                    except Exception:
                        pass

                    # also ensure payload contains thscode/symbol if it is a dict (provider adapters often rely on it)
                    try:
                        pld = getattr(pr, "payload", None)
                        if isinstance(pld, dict):
                            pld.setdefault("thscode", symbol)
                            pld.setdefault("symbol", symbol)
                    except Exception:
                        pass

                    _rid, created = repo.data_requests.enqueue_planned(pr, provider=settings.DATA_PROVIDER)
                    if created:
                        planned += 1

        s.commit()
        res["planned_requests"] = planned

    return {
        "trading_day": datetime.strptime(td, "%Y%m%d").strftime("%Y-%m-%d"),
        "target_day": datetime.strptime(target_td, "%Y%m%d").strftime("%Y-%m-%d"),
        **res,
    }


@router.get("/ui/watchlist")
def ui_watchlist(limit: int = 200) -> list[dict]:
    with SessionLocal() as s:
        repo = Repo(s)
        rows = repo.watchlist.list(limit=limit)
        s.commit()
        out: list[dict] = []
        for r in rows:
            out.append(
                {
                    "symbol": normalize_symbol(r.symbol),
                    "first_seen_day": r.first_seen_day,
                    "last_seen_day": r.last_seen_day,
                    "hit_count": int(r.hit_count or 0),
                    "active": bool(r.active),
                    "next_refresh_at": r.next_refresh_at.isoformat() if r.next_refresh_at else None,
                    "planner_state": r.planner_state,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
            )
        return out


@router.patch("/ui/watchlist/{symbol}")
def ui_watchlist_patch(symbol: str, payload: dict = Body(...)) -> dict:
    active = payload.get("active")
    if active is None:
        raise HTTPException(status_code=400, detail="active is required")
    with SessionLocal() as s:
        repo = Repo(s)
        row = repo.watchlist.set_active(symbol, bool(active))
        s.commit()
        return {
            "symbol": row.symbol,
            "active": bool(row.active),
            "hit_count": int(row.hit_count or 0),
            "next_refresh_at": row.next_refresh_at.isoformat() if row.next_refresh_at else None,
        }


@router.get("/ui/symbol/{symbol}/snapshots")
def ui_symbol_snapshots(symbol: str, limit: int = 50) -> list[dict]:
    symbol = normalize_symbol((symbol or "").strip())
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")
    with SessionLocal() as s:
        repo = Repo(s)
        rows = repo.feature_snapshots.list_by_symbol(symbol, limit=limit)
        s.commit()
        return [
            {
                "snapshot_id": r.snapshot_id,
                "symbol": normalize_symbol(r.symbol),
                "feature_set": r.feature_set,
                "asof_ts": r.asof_ts.isoformat(),
                "planner_version": r.planner_version,
                "request_ids": r.request_ids,
                "features": r.features,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


@router.get("/ui/labeling_settings")
def ui_labeling_settings() -> dict:
    # read-only view of pipeline knobs (controlled via env)
    return {
        "auto_fetch_enabled": bool(settings.LABELING_AUTO_FETCH_ENABLED),
        "poll_ms": int(settings.LABELING_PIPELINE_POLL_MS),
        "refresh_base_sec": int(settings.LABELING_REFRESH_BASE_SEC),
        "refresh_active_sec": int(settings.LABELING_REFRESH_ACTIVE_SEC),
        "history_days_base": int(settings.LABELING_HISTORY_DAYS_BASE),
        "history_days_expand": int(settings.LABELING_HISTORY_DAYS_EXPAND),
        "history_days_max": int(settings.LABELING_HISTORY_DAYS_MAX),
        "hf_limit_base": int(settings.LABELING_HF_LIMIT_BASE),
        "max_symbols_per_cycle": int(settings.LABELING_MAX_SYMBOLS_PER_CYCLE),
    }


@router.get("/accounts")
def list_accounts() -> list[dict]:
    with SessionLocal() as s:
        repo = Repo(s)
        repo.accounts.ensure_accounts_seeded()
        rows = repo.accounts.list_accounts()
        s.commit()
        return [{"account_id": r.account_id, "broker_type": r.broker_type, "created_at": r.created_at.isoformat()} for r in rows]

@router.get("/recommendations/trace")
def get_recommendation_trace(symbol: str, limit_days: int = 10) -> dict:
    """Trace a symbol's recommendation changes over recent trading days.

    Returns v2 data if available; falls back to legacy decisions otherwise.
    """
    sym = normalize_symbol(symbol)
    limit_days = max(1, min(int(limit_days), 60))

    with SessionLocal() as s:
        # Prefer v2
        q = (
            select(models.ModelRecommendationV2, models.ModelRunV2)
            .join(models.ModelRunV2, models.ModelRunV2.run_id == models.ModelRecommendationV2.run_id)
            .where(models.ModelRecommendationV2.symbol == sym)
            .order_by(models.ModelRecommendationV2.decision_day.desc())
            .limit(limit_days)
        )
        rows = s.execute(q).all()

        if rows:
            recos: list[dict] = []
            reco_ids = [r[0].reco_id for r in rows]
            ev_rows = s.execute(
                select(models.ModelRecommendationEvidenceV2)
                .where(models.ModelRecommendationEvidenceV2.reco_id.in_(reco_ids))
                .order_by(models.ModelRecommendationEvidenceV2.reco_id.asc(), models.ModelRecommendationEvidenceV2.importance.desc())
            ).scalars().all()
            ev_map: dict[int, list[dict]] = {}
            for ev in ev_rows:
                ev_map.setdefault(ev.reco_id, []).append(
                    {
                        "reason_code": ev.reason_code,
                        "reason_text": ev.reason_text,
                        "importance": int(ev.importance or 0),
                        "evidence": ev.evidence or {},
                        "refs": ev.refs or {},
                    }
                )

            for reco, run in rows:
                recos.append(
                    {
                        "decision_day": reco.decision_day,
                        "cutoff_ts": reco.cutoff_ts.isoformat() if reco.cutoff_ts else None,
                        "run": {
                            "model_name": run.model_name,
                            "model_version": run.model_version,
                            "objective": run.objective,
                            "horizon_days": int(run.horizon_days),
                            "target_profit_low": float(run.target_profit_low),
                            "target_profit_high": float(run.target_profit_high),
                            "params": run.params or {},
                            "label_version": run.label_version,
                            "created_at": run.created_at.isoformat() if run.created_at else None,
                        },
                        "recommendation": {
                            "action": reco.action,
                            "score": float(reco.score),
                            "confidence": float(reco.confidence),
                            "signals": reco.signals or {},
                            "lineage": reco.lineage or {},
                            "created_at": reco.created_at.isoformat() if reco.created_at else None,
                        },
                        "evidence": ev_map.get(reco.reco_id, []),
                    }
                )

            return {"symbol": sym, "mode": "v2", "trace": recos}

        # Fallback: legacy decisions
        q2 = (
            select(models.ModelDecision)
            .where(models.ModelDecision.symbol == sym)
            .order_by(models.ModelDecision.decision_time.desc())
            .limit(limit_days)
        )
        legacy = s.execute(q2).scalars().all()
        if not legacy:
            return {"symbol": sym, "mode": "none", "trace": []}

        ids = [d.id for d in legacy]
        ev_rows2 = s.execute(
            select(models.DecisionEvidence)
            .where(models.DecisionEvidence.decision_id.in_(ids))
            .order_by(models.DecisionEvidence.decision_id.asc(), models.DecisionEvidence.importance.desc())
        ).scalars().all()
        ev_map2: dict[int, list[dict]] = {}
        for ev in ev_rows2:
            ev_map2.setdefault(ev.decision_id, []).append(
                {
                    "reason_code": ev.reason_code,
                    "reason_text": ev.reason_text,
                    "importance": int(ev.importance or 0),
                    "evidence": ev.evidence_data or {},
                }
            )

        trace = []
        for d in legacy:
            trace.append(
                {
                    "decision_time": d.decision_time.isoformat() if d.decision_time else None,
                    "model_name": d.model_name,
                    "model_version": d.model_version,
                    "action": d.action,
                    "score": float(d.score),
                    "confidence": float(d.confidence),
                    "evidence": ev_map2.get(d.id, []),
                }
            )

        return {"symbol": sym, "mode": "legacy", "trace": trace}

@router.post("/tasks/eod_1530")
def task_eod_1530(trading_day: str | None = None) -> dict:
    """Run 15:30 (Asia/Shanghai) cutoff materialization.

    This endpoint is designed to be called by an external scheduler.
    """
    res = run_eod_cutoff_1530(trading_day=trading_day)
    return {"ok": True, **res}
