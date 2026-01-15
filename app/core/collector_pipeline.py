from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.adapters.data_provider import get_data_provider
from app.config import settings
from app.database import models
from app.utils.time import now_shanghai
from app.utils.trading_calendar import iter_prev_trading_days


def _step_get_or_create(session, batch_id: str, step_name: str) -> models.PipelineStep:
    row = (
        session.execute(
            select(models.PipelineStep).where(models.PipelineStep.batch_id == batch_id).where(models.PipelineStep.step_name == step_name)
        )
        .scalars()
        .first()
    )
    if row:
        return row
    row = models.PipelineStep(batch_id=batch_id, step_name=step_name, status="PENDING", detail={}, updated_at=now_shanghai())
    session.add(row)
    session.flush()
    return row


def _mark_step(session, step: models.PipelineStep, status: str, detail: dict[str, Any]) -> None:
    step.status = status
    step.detail = detail
    step.updated_at = now_shanghai()
    session.flush()


@dataclass
class CollectorResult:
    status: str  # DONE/FAILED/SKIPPED
    detail: dict[str, Any]


def collect_history_eod(session, *, batch_id: str, trading_day: str, symbols: list[str]) -> CollectorResult:
    """Collect historical EOD bars (offline-capable).

    This is intentionally designed to run on any day (non-trading day included),
    since historical data can be fetched/filled at any time.

    Provider:
    - MockDataProvider (default) returns plausible OHLCV without dates.
    - IFIND_HTTP provider can be enabled by setting DATA_PROVIDER=IFIND_HTTP
      and providing IFIND_HTTP_* tokens.

    Storage:
    - equity_eod_snapshot, keyed by (trading_day, symbol)
    """
    lookback = int(getattr(settings, "HISTORY_LOOKBACK_DAYS", 60) or 60)
    provider = get_data_provider()

    days = iter_prev_trading_days(trading_day, lookback)
    if not days:
        return CollectorResult(status="SKIPPED", detail={"reason": "no_trading_days_resolved"})

    upserted = 0
    errors: list[str] = []

    for sym in symbols:
        try:
            payload = {
                # Mock provider uses `symbol`; real iFinD may use `codes`.
                "symbol": sym,
                "codes": sym,
                "indicators": "open,high,low,close,volume,amount",
                "limit": len(days),
            }
            resp = provider.call("cmd_history_quotation", payload)
            raw = resp.raw if isinstance(resp.raw, dict) else {}
            tables = raw.get("tables")
            table = None
            if isinstance(tables, list) and tables:
                t0 = tables[0]
                if isinstance(t0, dict):
                    table = t0.get("table")

            if not isinstance(table, dict):
                # Nothing usable
                continue

            opens = table.get("open") or []
            highs = table.get("high") or []
            lows = table.get("low") or []
            closes = table.get("close") or []
            vols = table.get("volume") or []
            amts = table.get("amount") or []

            # Align lengths; use min to avoid index errors.
            n = min(len(days), len(opens), len(highs), len(lows), len(closes), len(vols) if isinstance(vols, list) else len(days))
            if n <= 0:
                continue

            # Most APIs return oldest->newest; our mock does too.
            # Map last n items to the last n days.
            days_used = list(reversed(days))[-n:]
            idx0 = len(opens) - n

            for i in range(n):
                d = days_used[i]
                o = opens[idx0 + i] if idx0 + i < len(opens) else None
                h = highs[idx0 + i] if idx0 + i < len(highs) else None
                l = lows[idx0 + i] if idx0 + i < len(lows) else None
                c = closes[idx0 + i] if idx0 + i < len(closes) else None
                v = vols[idx0 + i] if isinstance(vols, list) and idx0 + i < len(vols) else None
                a = amts[idx0 + i] if isinstance(amts, list) and idx0 + i < len(amts) else None

                row = models.EquityEODSnapshot(
                    trading_day=d,
                    symbol=sym,
                    prev_close=None,
                    open=float(o) if o is not None else None,
                    high=float(h) if h is not None else None,
                    low=float(l) if l is not None else None,
                    close=float(c) if c is not None else None,
                    volume=float(v) if v is not None else None,
                    amount=float(a) if a is not None else None,
                    turnover_rate=None,
                    amplitude=None,
                    float_market_cap=None,
                    is_limit_up_close=None,
                    source=str(getattr(settings, "DATA_PROVIDER", "MOCK") or "MOCK"),
                    raw_ref=str(resp.payload_sha256),
                    updated_at=now_shanghai(),
                )

                session.merge(row)
                upserted += 1

        except Exception as e:
            errors.append(f"{sym}:{type(e).__name__}")
            continue

    detail = {"upserted": upserted, "symbols": len(symbols), "lookback": lookback}
    if errors:
        detail["errors"] = errors[:20]
    return CollectorResult(status="DONE" if upserted > 0 else "SKIPPED", detail=detail)


def collect_theme_and_sector(session, *, batch_id: str, trading_day: str, symbols: list[str]) -> CollectorResult:
    """Collect theme/sector mapping & stats (offline-capable).

    Real theme data is provider-specific and often requires paid endpoints.
    This step is still part of the pipeline (so schema & observability exist),
    but by default it will SKIP unless a provider config is supplied.

    Config expected (optional):
    - THEME_PROVIDER: IFIND_HTTP/MOCK
    - THEME_ENDPOINT_MAP: endpoint name that returns a mapping for a symbol
    - THEME_ENDPOINT_STATS: endpoint name that returns theme stats (optional)
    """
    theme_provider = (getattr(settings, "THEME_PROVIDER", "") or "").strip().upper()
    endpoint_map = (getattr(settings, "THEME_ENDPOINT_MAP", "") or "").strip()

    if not theme_provider or not endpoint_map:
        return CollectorResult(status="SKIPPED", detail={"reason": "theme_provider_not_configured"})

    provider = get_data_provider()
    upserted = 0
    errors: list[str] = []

    for sym in symbols:
        try:
            resp = provider.call(endpoint_map, {"symbol": sym, "codes": sym, "date": trading_day})
            raw = resp.raw if isinstance(resp.raw, dict) else {}
            items = raw.get("items") or raw.get("data") or []
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                theme_id = str(it.get("theme_id") or it.get("id") or it.get("code") or "").strip()
                if not theme_id:
                    continue
                theme_name = str(it.get("theme_name") or it.get("name") or "").strip()
                theme_rank = it.get("rank")
                row = models.EquityThemeMap(
                    trading_day=trading_day,
                    symbol=sym,
                    theme_id=theme_id,
                    theme_name=theme_name,
                    theme_rank=int(theme_rank) if isinstance(theme_rank, (int, float)) else None,
                    source=theme_provider,
                    raw_ref=str(resp.payload_sha256),
                    updated_at=now_shanghai(),
                )
                session.merge(row)
                upserted += 1
        except Exception as e:
            errors.append(f"{sym}:{type(e).__name__}")

    detail = {"upserted": upserted, "symbols": len(symbols)}
    if errors:
        detail["errors"] = errors[:20]
    return CollectorResult(status="DONE" if upserted > 0 else "SKIPPED", detail=detail)


def run_collectors_for_committed_batch(session, batch: models.LimitupPoolBatch) -> None:
    """Run offline collectors for a committed batch.

    This runs idempotently (each step once per batch).
    """
    trading_day = batch.trading_day
    batch_id = batch.batch_id

    # Only collect for candidates that are READY (p_limit_up already provided), since
    # that's the operator-confirmed set.
    syms = (
        session.execute(
            select(models.LimitupCandidate.symbol)
            .where(models.LimitupCandidate.batch_id == batch_id)
            .where(models.LimitupCandidate.candidate_status == "READY")
            .order_by(models.LimitupCandidate.symbol.asc())
        )
        .scalars()
        .all()
    )
    symbols = [str(x) for x in syms if isinstance(x, str) and x]

    # Step 1: history (offline)
    s1 = _step_get_or_create(session, batch_id, "collector.history_eod")
    if s1.status not in {"DONE"}:
        _mark_step(session, s1, "RUNNING", {"started_at": now_shanghai().isoformat()})
        r1 = collect_history_eod(session, batch_id=batch_id, trading_day=trading_day, symbols=symbols)
        _mark_step(session, s1, r1.status if r1.status != "SKIPPED" else "DONE", r1.detail)

    # Step 2: theme/sector (offline-capable, usually needs provider)
    s2 = _step_get_or_create(session, batch_id, "collector.theme")
    if s2.status not in {"DONE"}:
        _mark_step(session, s2, "RUNNING", {"started_at": now_shanghai().isoformat()})
        r2 = collect_theme_and_sector(session, batch_id=batch_id, trading_day=trading_day, symbols=symbols)
        _mark_step(session, s2, r2.status if r2.status != "SKIPPED" else "DONE", r2.detail)
