"""End-of-day cutoff tasks (15:30 Beijing time).

This module intentionally does NOT fetch data from external providers.
It assumes raw payloads and/or labeling candidates were already ingested.

At 15:30 the system typically:
1) Ensures high-frequency facts for the day are ingested (1m bars, ticks, L1 orderbook snapshots).
2) Materializes a feature snapshot `FeatureIntradayCutoffV2` for each symbol.
3) Optionally writes `OnlineFeedbackEventV2` for label generation when forward data is available.

External schedulers can call the HTTP endpoint that wraps these functions.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.database.engine import SessionLocal
from app.database import models


def _bj_now_iso() -> str:
    # We keep internal timestamps as epoch/int/UTC where appropriate.
    # Display formatting is handled on the frontend; however settings.TZ is Asia/Shanghai.
    return datetime.now(tz=timezone.utc).isoformat()


def materialize_feat_intraday_cutoff(
    trading_day: str,
    cutoff_ts: Optional[int] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute and upsert `FeatureIntradayCutoffV2` rows.

    Parameters:
    - trading_day: 'YYYY-MM-DD' in Asia/Shanghai calendar
    - cutoff_ts: unix seconds; if not provided, defaults to 'trading_day 15:30' (Asia/Shanghai) converted to epoch.
    - symbol: optional single symbol filter

    Returns summary counters.
    """

    # If caller doesn't pass cutoff_ts, we store a sentinel 15:30 local time encoded as "trading_day".
    # Downstream can treat cutoff_ts=0 as "15:30 local" until a trading calendar module is added.
    cutoff_ts = int(cutoff_ts or 0)

    with SessionLocal() as s:
        q = s.query(models.LabelingCandidate).filter(models.LabelingCandidate.trading_day == trading_day)
        if symbol:
            q = q.filter(models.LabelingCandidate.symbol == symbol)
        candidates: List[models.LabelingCandidate] = q.all()

        upserted = 0
        for c in candidates:
            feats = {
                "p_limit_up": c.p_limit_up,
                "price": c.price,
                "turnover_rate": c.turnover_rate,
                "order_amount": c.order_amount,
                "volume": c.volume,
                "amount": c.amount,
                "high_days": c.high_days,
                "is_again_limit": c.is_again_limit,
                "is_new": c.is_new,
                "is_yizi": c.is_yizi,
                "limit_up_type": c.limit_up_type,
                "limit_up_reason": c.limit_up_reason,
                "src": "labeling_candidate",
                "materialized_at_utc": _bj_now_iso(),
            }

            existing = (
                s.query(models.FeatureIntradayCutoffV2)
                .filter(
                    models.FeatureIntradayCutoffV2.symbol == c.symbol,
                    models.FeatureIntradayCutoffV2.trading_day == trading_day,
                    models.FeatureIntradayCutoffV2.cutoff_ts == cutoff_ts,
                )
                .one_or_none()
            )

            if existing:
                existing.features = feats
            else:
                s.add(
                    models.FeatureIntradayCutoffV2(
                        symbol=c.symbol,
                        trading_day=trading_day,
                        cutoff_ts=cutoff_ts,
                        features=feats,
                        created_at=int(datetime.now(tz=timezone.utc).timestamp()),
                    )
                )

            upserted += 1

        s.commit()

    return {
        "trading_day": trading_day,
        "cutoff_ts": cutoff_ts,
        "symbol": symbol,
        "candidates": len(candidates),
        "upserted": upserted,
    }
