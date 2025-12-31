from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models
from app.utils.time import now_shanghai, trading_day_str
from app.utils.crypto import sha256_hex


@dataclass
class SystemEventsRepo:
    s: Session

    def write_event(
        self,
        event_type: str,
        correlation_id: str | None,
        severity: str,
        payload: dict,
        symbol: str | None = None,
    ) -> None:
        self.s.add(
            models.SystemEvent(
                event_type=event_type,
                severity=severity,
                correlation_id=correlation_id,
                symbol=symbol,
                payload=payload,
                time=now_shanghai(),
            )
        )


@dataclass
class SystemStatusRepo:
    s: Session

    def get_for_update(self) -> models.SystemStatus:
        row = self.s.execute(select(models.SystemStatus).where(models.SystemStatus.id == 1)).scalar_one_or_none()
        if row is None:
            row = models.SystemStatus(id=1, updated_at=now_shanghai())
            self.s.add(row)
            self.s.flush()
        return row

    def set_panic_halt(self, veto_code: str) -> None:
        st = self.get_for_update()
        st.panic_halt = True
        st.guard_level = 3
        st.veto = True
        st.veto_code = veto_code
        st.updated_at = now_shanghai()

    def set_challenge(self, code: str | None) -> None:
        st = self.get_for_update()
        st.challenge_code = code
        st.updated_at = now_shanghai()

    def set_self_check(self, report_hash: str) -> None:
        st = self.get_for_update()
        st.last_self_check_report_hash = report_hash
        st.last_self_check_time = now_shanghai()
        st.updated_at = now_shanghai()

    def reset_from_panic(self) -> None:
        st = self.get_for_update()
        st.panic_halt = False
        st.veto = False
        st.veto_code = ""
        st.guard_level = 0
        st.challenge_code = None
        st.updated_at = now_shanghai()


@dataclass
class NonceRepo:
    s: Session

    def next_nonce(self, symbol: str, strategy_id: str) -> int:
        td = trading_day_str(now_shanghai())
        row = (
            self.s.execute(
                select(models.NonceCursor)
                .where(
                    models.NonceCursor.trading_day == td,
                    models.NonceCursor.symbol == symbol,
                    models.NonceCursor.strategy_id == strategy_id,
                )
                .with_for_update()
            )
            .scalar_one_or_none()
        )
        if row is None:
            row = models.NonceCursor(
                trading_day=td,
                symbol=symbol,
                strategy_id=strategy_id,
                last_nonce=0,
                updated_at=now_shanghai(),
            )
            self.s.add(row)
            self.s.flush()
        row.last_nonce = int(row.last_nonce) + 1
        row.updated_at = now_shanghai()
        return int(row.last_nonce)


@dataclass
class SymbolLockRepo:
    s: Session

    def is_locked(self, symbol: str) -> bool:
        row = self.s.get(models.SymbolLock, symbol)
        return bool(row.locked) if row else False

    def lock(self, symbol: str, reason: str, ref: str | None) -> None:
        now = now_shanghai()
        row = self.s.get(models.SymbolLock, symbol)
        if row is None:
            row = models.SymbolLock(
                symbol=symbol,
                locked=True,
                lock_reason=reason,
                lock_ref=ref,
                created_at=now,
                updated_at=now,
            )
            self.s.add(row)
        else:
            row.locked = True
            row.lock_reason = reason
            row.lock_ref = ref
            row.updated_at = now

    def unlock(self, symbol: str) -> None:
        row = self.s.get(models.SymbolLock, symbol)
        if row:
            row.locked = False
            row.lock_reason = ""
            row.lock_ref = None
            row.updated_at = now_shanghai()


@dataclass
class OutboxRepo:
    s: Session

    def enqueue(self, event_type: str, dedupe_key: str, payload: dict) -> None:
        exists = self.s.execute(select(models.OutboxEvent).where(models.OutboxEvent.dedupe_key == dedupe_key)).scalar_one_or_none()
        if exists is not None:
            return
        now = now_shanghai()
        self.s.add(
            models.OutboxEvent(
                event_type=event_type,
                dedupe_key=dedupe_key,
                status="PENDING",
                attempts=0,
                available_at=now,
                payload=payload,
                created_at=now,
                sent_at=None,
            )
        )

    def fetch_pending(self, limit: int = 50) -> list[models.OutboxEvent]:
        now = now_shanghai()
        return (
            self.s.execute(
                select(models.OutboxEvent)
                .where(
                    models.OutboxEvent.status == "PENDING",
                    models.OutboxEvent.available_at <= now,
                )
                .order_by(models.OutboxEvent.id.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )

    def mark_sent(self, ev: models.OutboxEvent) -> None:
        ev.status = "SENT"
        ev.sent_at = now_shanghai()
        ev.last_error = None

    def _backoff_ms(self, attempts: int) -> int:
        # deterministic exponential backoff with cap
        base = int(settings.OUTBOX_BACKOFF_BASE_MS)
        cap = int(settings.OUTBOX_BACKOFF_MAX_MS)
        # attempts starts at 1 on first failure
        ms = base * (2 ** max(0, attempts - 1))
        return int(min(cap, ms))

    def mark_failed(self, ev: models.OutboxEvent, err: str) -> None:
        ev.attempts = int(ev.attempts) + 1
        ev.last_error = (err or "")[:500]

        if int(ev.attempts) >= int(settings.OUTBOX_MAX_ATTEMPTS):
            ev.status = "DEAD"
            ev.available_at = now_shanghai()
            return

        wait_ms = self._backoff_ms(int(ev.attempts))
        ev.status = "PENDING"
        ev.available_at = now_shanghai() + timedelta(milliseconds=wait_ms)


@dataclass
class OrderAnchorRepo:
    s: Session

    def upsert_anchor(
        self,
        cid: str,
        client_order_id: str,
        broker_order_id: str | None,
        request_uuid: str,
        ack_hash: str,
        raw_request_hash: str,
        raw_response_hash: str,
    ) -> None:
        row = self.s.get(models.OrderAnchor, cid)
        if row is None:
            row = models.OrderAnchor(
                cid=cid,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                request_uuid=request_uuid,
                ack_hash=ack_hash,
                raw_request_hash=raw_request_hash,
                raw_response_hash=raw_response_hash,
                created_at=now_shanghai(),
            )
            self.s.add(row)
        else:
            row.broker_order_id = broker_order_id
            row.ack_hash = ack_hash
            row.raw_request_hash = raw_request_hash
            row.raw_response_hash = raw_response_hash


@dataclass
class FrozenVersionsRepo:
    s: Session

    def ensure_today_frozen(
        self,
        rule_set_version_hash: str,
        strategy_contract_hash: str,
        model_snapshot_uuid: str,
        cost_model_version: str,
        canonicalization_version: str,
        feature_extractor_version: str,
    ) -> None:
        td = trading_day_str(now_shanghai())
        row = self.s.get(models.DailyFrozenVersions, td)
        report = {
            "trading_day": td,
            "rule_set_version_hash": rule_set_version_hash,
            "strategy_contract_hash": strategy_contract_hash,
            "model_snapshot_uuid": model_snapshot_uuid,
            "cost_model_version": cost_model_version,
            "canonicalization_version": canonicalization_version,
            "feature_extractor_version": feature_extractor_version,
        }
        report_hash = sha256_hex(str(report).encode("utf-8"))

        if row is None:
            self.s.add(
                models.DailyFrozenVersions(
                    trading_day=td,
                    rule_set_version_hash=rule_set_version_hash,
                    strategy_contract_hash=strategy_contract_hash,
                    model_snapshot_uuid=model_snapshot_uuid,
                    cost_model_version=cost_model_version,
                    canonicalization_version=canonicalization_version,
                    feature_extractor_version=feature_extractor_version,
                    report_hash=report_hash,
                    created_at=now_shanghai(),
                )
            )
        else:
            if row.report_hash != report_hash:
                raise ValueError("daily_frozen_versions_mismatch")


@dataclass
class Repo:
    s: Session

    @property
    def system_events(self) -> SystemEventsRepo:
        return SystemEventsRepo(self.s)

    @property
    def system_status(self) -> SystemStatusRepo:
        return SystemStatusRepo(self.s)

    @property
    def nonce(self) -> NonceRepo:
        return NonceRepo(self.s)

    @property
    def symbol_lock(self) -> SymbolLockRepo:
        return SymbolLockRepo(self.s)

    @property
    def outbox(self) -> OutboxRepo:
        return OutboxRepo(self.s)

    @property
    def anchors(self) -> OrderAnchorRepo:
        return OrderAnchorRepo(self.s)

    @property
    def frozen_versions(self) -> FrozenVersionsRepo:
        return FrozenVersionsRepo(self.s)
