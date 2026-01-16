from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.data_provider import get_data_provider
from app.config import settings
from app.database import models
from app.utils.trading_calendar import trading_days_between, next_n_trading_days
from app.utils.calendar_backfill import ensure_trading_calendar
from app.utils.time import now_shanghai
from app.utils.crypto import sha256_hex
from app.core.model_training import ensure_labels_for_candidates, train_objective_lightgbm, OBJ_TP5_3D, OBJ_TP8_3D
from app.utils.symbols import normalize_symbol


def _parse_day(s: str) -> date:
    s = (s or "").strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s, "%Y-%m-%d").date()


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _extract_table(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    tables = raw.get("tables")
    if isinstance(tables, list) and tables:
        t0 = tables[0]
        if isinstance(t0, dict):
            tab = t0.get("table")
            if isinstance(tab, dict):
                return tab
    return None


def _to_dec(x: Any) -> Optional[Decimal]:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except Exception:
        return None


@dataclass
class BackfillStats:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0


def backfill_daily_ohlcv(
    session: Session,
    *,
    symbols: list[str],
    start_day: date,
    end_day: date,
    source: str,
) -> BackfillStats:
    """Backfill FactDailyOHLCV for symbols and [start_day,end_day] (trading days).

    This uses adapters.data_provider.get_data_provider() and expects the provider
    to support an iFinD-like endpoint shape: cmd_history_quotation.

    NOTE: For providers that return explicit dates, we prefer those dates.
    Otherwise we align returned arrays to the resolved trading day list.
    """

    provider = get_data_provider()

    days = trading_days_between(start_day, end_day, session=session)
    if not days:
        return BackfillStats(skipped=len(symbols))

    # Preload existing keys
    q = select(models.FactDailyOHLCV.instrument_id, models.FactDailyOHLCV.trading_day).where(
        models.FactDailyOHLCV.instrument_id.in_([normalize_symbol(s) for s in symbols]),
        models.FactDailyOHLCV.trading_day.in_(days),
    )
    existing = set(session.execute(q).all())

    stats = BackfillStats()

    for sym in symbols:
        inst = normalize_symbol(sym)
        try:
            payload = {
                "symbol": inst,
                "codes": inst,
                "indicators": "open,high,low,close,volume,amount",
                "startdate": _yyyymmdd(start_day),
                "enddate": _yyyymmdd(end_day),
                "limit": len(days),
            }
            resp = provider.call("cmd_history_quotation", payload)
            raw = resp.raw if isinstance(resp.raw, dict) else {}
            tab = _extract_table(raw)
            if not isinstance(tab, dict):
                stats.errors += 1
                continue

            # Prefer explicit date list if provided
            dates = tab.get("date") or tab.get("dates") or tab.get("trade_date")
            if isinstance(dates, list) and dates:
                day_list: list[date] = []
                for ds in dates:
                    try:
                        ds2 = str(ds).replace("-", "")
                        if len(ds2) == 8 and ds2.isdigit():
                            day_list.append(datetime.strptime(ds2, "%Y%m%d").date())
                    except Exception:
                        continue
            else:
                day_list = days

            opens = tab.get("open") or []
            highs = tab.get("high") or []
            lows = tab.get("low") or []
            closes = tab.get("close") or []
            vols = tab.get("volume") or tab.get("vol") or []
            amts = tab.get("amount") or []

            # Align lengths
            n = min(len(day_list), len(opens), len(highs), len(lows), len(closes))
            if n <= 0:
                stats.errors += 1
                continue

            # Most providers return oldest->newest; we assume so.
            # Keep the last n items if provider returns longer series.
            idx0 = max(0, len(opens) - n)
            day_used = day_list[-n:]

            for i in range(n):
                d = day_used[i]
                key = (inst, d)
                o = _to_dec(opens[idx0 + i])
                h = _to_dec(highs[idx0 + i])
                l = _to_dec(lows[idx0 + i])
                c = _to_dec(closes[idx0 + i])
                if o is None or h is None or l is None or c is None:
                    stats.skipped += 1
                    continue

                vol = None
                amt = None
                if isinstance(vols, list) and idx0 + i < len(vols):
                    vol = _to_dec(vols[idx0 + i])
                if isinstance(amts, list) and idx0 + i < len(amts):
                    amt = _to_dec(amts[idx0 + i])

                row = session.execute(
                    select(models.FactDailyOHLCV).where(
                        models.FactDailyOHLCV.instrument_id == inst,
                        models.FactDailyOHLCV.trading_day == d,
                    )
                ).scalar_one_or_none()

                if row is None:
                    session.add(
                        models.FactDailyOHLCV(
                            instrument_id=inst,
                            trading_day=d,
                            open=o,
                            high=h,
                            low=l,
                            close=c,
                            volume=vol,
                            amount=amt,
                            source=source,
                            raw_hash=resp.payload_sha256,
                        )
                    )
                    stats.inserted += 1
                else:
                    row.open = o
                    row.high = h
                    row.low = l
                    row.close = c
                    row.volume = vol
                    row.amount = amt
                    row.source = source
                    row.raw_hash = resp.payload_sha256
                    stats.updated += 1

                existing.add(key)

        except Exception:
            stats.errors += 1
            continue

    return stats


@dataclass
class PipelineResult:
    trading_days: list[str]
    backfill: dict[str, Any]
    labels: dict[str, Any]
    train: dict[str, Any]
    activated: dict[str, Any]
    started_ts: str
    finished_ts: str


def run_ml_pipeline(
    session: Session,
    *,
    start_day: str,
    end_day: str,
    label_version: str | None = None,
    do_backfill: bool = True,
    do_labels: bool = True,
    do_train: bool = True,
    max_symbols_per_day: int = 5000,
    max_rows: int = 5000,
) -> PipelineResult:
    """One-key pipeline:

    - Resolve trading days in [start,end]
    - (optional) backfill fact_daily_ohlcv for candidate symbols and the days needed for labels
    - (optional) generate labels for each day
    - (optional) train & activate LightGBM TP5_3D / TP8_3D

    This function is synchronous and intended for admin/manual runs.
    """

    t0 = now_shanghai()

    sday = _parse_day(start_day)
    eday = _parse_day(end_day)

    # Ensure trading calendar table is populated for the requested range.
    # We include a small buffer so T+1..T+3 resolution does not fall back to
    # weekend-only logic when the table is empty.
    try:
        from datetime import timedelta

        buf_start = min(sday, eday) - timedelta(days=14)
        buf_end = max(sday, eday) + timedelta(days=14)
        ensure_trading_calendar(
            session,
            start_day=buf_start,
            end_day=buf_end,
            market_code=(getattr(settings, "MARKET_CALENDAR_CODE", None) or "XSHG"),
            note="computed_by_pandas_market_calendars",
        )
        # SessionLocal uses autoflush=False; flush so subsequent selects can see new rows.
        session.flush()
    except Exception:
        # If calendar computation fails for any reason, keep existing behavior:
        # fall back to weekend-only logic.
        pass

    tdays = trading_days_between(sday, eday, session=session)
    tdays_str = [_yyyymmdd(d) for d in tdays]

    lv = (label_version or settings.LABEL3D_VERSION or "v1").strip()

    backfill_total = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
    labels_total = {"labels_upserted": 0, "days_processed": 0}

    if do_backfill or do_labels:
        for d in tdays:
            td = _yyyymmdd(d)
            # candidates for this day
            cands = (
                session.execute(
                    select(models.LabelingCandidate.symbol).where(models.LabelingCandidate.trading_day == td).limit(max_symbols_per_day)
                )
                .scalars()
                .all()
            )
            symbols = [normalize_symbol(x) for x in cands if x]

            if do_backfill and symbols:
                # For each T we need daily data for T (gap) and T+1..T+3 (labels)
                future = next_n_trading_days(d, 3, session=session)
                if len(future) == 3:
                    days_needed = [d] + future
                    bf = backfill_daily_ohlcv(
                        session,
                        symbols=symbols,
                        start_day=min(days_needed),
                        end_day=max(days_needed),
                        source=(settings.DATA_PROVIDER or "UNKNOWN").upper(),
                    )
                    backfill_total["inserted"] += bf.inserted
                    backfill_total["updated"] += bf.updated
                    backfill_total["skipped"] += bf.skipped
                    backfill_total["errors"] += bf.errors

            if do_labels:
                cnt = ensure_labels_for_candidates(session, trading_day=td, label_version=lv, max_symbols=max_symbols_per_day)
                labels_total["labels_upserted"] += int(cnt)
                labels_total["days_processed"] += 1

    train_out: dict[str, Any] = {"trained": []}
    activated: dict[str, Any] = {}
    if do_train:
        r5 = train_objective_lightgbm(session, OBJ_TP5_3D, label_version=lv, max_rows=int(max_rows))
        r8 = train_objective_lightgbm(session, OBJ_TP8_3D, label_version=lv, max_rows=int(max_rows))
        train_out = {
            "trained": [
                {"objective": r5.objective, "model_version": r5.model_version, "metrics": r5.metrics, "artifact_sha256": r5.artifact_sha256},
                {"objective": r8.objective, "model_version": r8.model_version, "metrics": r8.metrics, "artifact_sha256": r8.artifact_sha256},
            ]
        }
        activated = {"TP5_3D": r5.model_version, "TP8_3D": r8.model_version}

    t1 = now_shanghai()
    return PipelineResult(
        trading_days=tdays_str,
        backfill=backfill_total,
        labels=labels_total,
        train=train_out,
        activated=activated,
        started_ts=t0.isoformat(),
        finished_ts=t1.isoformat(),
    )
