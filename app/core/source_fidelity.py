from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models
from app.database.repo import Repo
from app.utils.time import now_shanghai


@dataclass
class FidelityInputs:
    symbol: str
    data_ts: object  # datetime
    channel_id_a: str
    channel_id_b: str
    close_a_int64: int
    close_b_int64: int
    tick_size: float


@dataclass
class FidelityResult:
    abs_diff: int
    threshold: int
    before: float
    after: float
    action_taken: str


class SourceFidelityEngine:
    """
    盘后跨源对账 + 评分 + 自动降级动作。
    """
    def evaluate_pair(self, s: Session, inp: FidelityInputs) -> FidelityResult:
        repo = Repo(s)

        abs_diff = abs(int(inp.close_a_int64) - int(inp.close_b_int64))
        tick_int = int(round(float(inp.tick_size) * 10000))
        threshold = max(tick_int, int(round(tick_int * float(settings.SOURCE_FIDELITY_K))))

        # mismatch => score down; match => score up slightly (bounded)
        cur_a = s.get(models.ChannelCursor, inp.channel_id_a)
        cur_b = s.get(models.ChannelCursor, inp.channel_id_b)

        # initialize cursors if missing (rare)
        if cur_a is None:
            cur_a = models.ChannelCursor(
                channel_id=inp.channel_id_a,
                last_seq=0,
                last_ingest_ts=now_shanghai(),
                quality_score=1.0,
                p99_latency_ms=settings.EPSILON_MIN_MS,
                p99_state={},
                fidelity_score=1.0,
                fidelity_low_streak=0,
                updated_at=now_shanghai(),
            )
            s.add(cur_a)
        if cur_b is None:
            cur_b = models.ChannelCursor(
                channel_id=inp.channel_id_b,
                last_seq=0,
                last_ingest_ts=now_shanghai(),
                quality_score=1.0,
                p99_latency_ms=settings.EPSILON_MIN_MS,
                p99_state={},
                fidelity_score=1.0,
                fidelity_low_streak=0,
                updated_at=now_shanghai(),
            )
            s.add(cur_b)
        s.flush()

        before = float(min(cur_a.fidelity_score, cur_b.fidelity_score))

        mismatch = abs_diff > threshold
        if mismatch:
            # penalize both channels a bit; keep >=0
            cur_a.fidelity_score = max(0.0, float(cur_a.fidelity_score) - 0.02)
            cur_b.fidelity_score = max(0.0, float(cur_b.fidelity_score) - 0.02)
        else:
            cur_a.fidelity_score = min(1.0, float(cur_a.fidelity_score) + 0.005)
            cur_b.fidelity_score = min(1.0, float(cur_b.fidelity_score) + 0.005)

        after = float(min(cur_a.fidelity_score, cur_b.fidelity_score))

        # action policy
        action = "NONE"
        if (before - after) >= float(settings.SOURCE_FIDELITY_DEGRADE_DELTA):
            action = "WEIGHT_DOWN"
            repo.system_events.write_event(
                event_type="SOURCE_FIDELITY_DEGRADED",
                correlation_id=None,
                severity="WARN",
                symbol=inp.symbol,
                payload={
                    "event_type": "SOURCE_FIDELITY_EVENT",
                    "symbol": inp.symbol,
                    "data_ts": inp.data_ts.isoformat(),
                    "channel_id_a": inp.channel_id_a,
                    "channel_id_b": inp.channel_id_b,
                    "metric": {"close_a": inp.close_a_int64, "close_b": inp.close_b_int64, "abs_diff": abs_diff, "threshold": threshold},
                    "fidelity_score_before": before,
                    "fidelity_score_after": after,
                    "action_taken": action,
                    "time": now_shanghai().isoformat(),
                },
            )
        else:
            repo.system_events.write_event(
                event_type="SOURCE_FIDELITY_EVENT",
                correlation_id=None,
                severity="INFO",
                symbol=inp.symbol,
                payload={
                    "event_type": "SOURCE_FIDELITY_EVENT",
                    "symbol": inp.symbol,
                    "data_ts": inp.data_ts.isoformat(),
                    "channel_id_a": inp.channel_id_a,
                    "channel_id_b": inp.channel_id_b,
                    "metric": {"close_a": inp.close_a_int64, "close_b": inp.close_b_int64, "abs_diff": abs_diff, "threshold": threshold},
                    "fidelity_score_before": before,
                    "fidelity_score_after": after,
                    "action_taken": action,
                    "time": now_shanghai().isoformat(),
                },
            )

        # low-score streak -> research-only policy marker (here: lower quality_score as proxy)
        for cur in (cur_a, cur_b):
            if float(cur.fidelity_score) < float(settings.SOURCE_FIDELITY_LOW_SCORE):
                cur.fidelity_low_streak = int(cur.fidelity_low_streak) + 1
            else:
                cur.fidelity_low_streak = 0

            if int(cur.fidelity_low_streak) >= int(settings.SOURCE_FIDELITY_LOW_STREAK_N):
                # hard degrade: push quality_score down (used by ingest + downstream)
                cur.quality_score = max(0.0, float(cur.quality_score) - 0.2)
                repo.system_events.write_event(
                    event_type="SOURCE_FIDELITY_RESEARCH_ONLY",
                    correlation_id=None,
                    severity="ERROR",
                    payload={"channel_id": cur.channel_id, "fidelity_score": cur.fidelity_score, "low_streak": cur.fidelity_low_streak},
                )

            cur.updated_at = now_shanghai()

        # persist daily audit row
        s.add(
            models.SourceFidelityDaily(
                symbol=inp.symbol,
                data_ts=inp.data_ts,
                channel_id_a=inp.channel_id_a,
                channel_id_b=inp.channel_id_b,
                close_a=int(inp.close_a_int64),
                close_b=int(inp.close_b_int64),
                abs_diff=int(abs_diff),
                threshold=int(threshold),
                fidelity_score_before=float(before),
                fidelity_score_after=float(after),
                action_taken=action,
                created_at=now_shanghai(),
            )
        )

        return FidelityResult(abs_diff=abs_diff, threshold=threshold, before=before, after=after, action_taken=action)
