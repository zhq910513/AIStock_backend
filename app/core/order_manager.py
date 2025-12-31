from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy import update, and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import models
from app.core.canonicalization import canonicalize_order
from app.database.repo import Repo
from app.utils.time import now_shanghai
from app.utils.ids import new_transition_id


ALL_STATES = {
    "CREATED", "SUBMITTED", "PENDING", "PARTIALLY_FILLED",
    "FILLED", "CANCELLED", "REJECTED", "EXPIRED",
    "AMBIGUOUS", "PANIC_HALT",
}


@dataclass
class SubmitResult:
    cid: str
    client_order_id: str
    state: str


class OrderManager:
    def _client_order_id(self, cid: str) -> str:
        return f"QEE-{cid}"

    def create_order(
        self,
        s: Session,
        cid: str,
        symbol: str,
        side: str,
        order_type: str,
        limit_price: float | None,
        qty: int,
        strategy_contract_hash: str,
    ) -> models.Order:
        canon = canonicalize_order(
            s=s,
            cid=cid,
            order_type=order_type,
            symbol=symbol,
            side=side,
            limit_price=limit_price,
            qty=qty,
        )

        now = now_shanghai()
        order = models.Order(
            cid=cid,
            client_order_id=self._client_order_id(cid),
            broker_order_id=None,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type.upper(),
            limit_price_int64=canon.limit_price_int64,
            qty_int=canon.qty_int,
            tick_rule_version=canon.tick_rule_version,
            lot_rule_version=canon.lot_rule_version,
            canonicalization_version=canon.canonicalization_version,
            metadata_hash=canon.metadata_hash,
            state="CREATED",
            version_id=1,
            last_transition_id=None,
            strategy_contract_hash=strategy_contract_hash,
            created_at=now,
            updated_at=now,
        )
        s.add(order)
        return order

    def transition(
        self,
        s: Session,
        cid: str,
        from_state: str,
        to_state: str,
        transition_id: str | None = None,
    ) -> None:
        if from_state not in ALL_STATES or to_state not in ALL_STATES:
            raise ValueError("Invalid state")

        repo = Repo(s)
        order = s.get(models.Order, cid)
        if order is None:
            raise ValueError("Order not found")

        if transition_id is None:
            transition_id = new_transition_id(cid, from_state, to_state, int(order.version_id))

        # Idempotent fast-path
        if order.state == to_state and order.last_transition_id == transition_id:
            return

        # Spec atomic gate:
        # WHERE cid=:cid AND version_id=:current_version AND last_transition_id!=:transition_id AND state=:from_state
        now = now_shanghai()
        stmt = (
            update(models.Order)
            .where(
                and_(
                    models.Order.cid == cid,
                    models.Order.version_id == int(order.version_id),
                    models.Order.state == from_state,
                    or_(models.Order.last_transition_id.is_(None), models.Order.last_transition_id != transition_id),
                )
            )
            .values(
                state=to_state,
                version_id=int(order.version_id) + 1,
                last_transition_id=transition_id,
                updated_at=now,
            )
        )
        res = s.execute(stmt)
        if res.rowcount != 1:
            # re-read to determine idempotent or conflict
            s.expire(order)
            order2 = s.get(models.Order, cid)
            if order2 and order2.state == to_state and order2.last_transition_id == transition_id:
                return
            raise ValueError("State transition conflict (optimistic lock / idempotency gate)")

        # record transition event (best-effort; unique constraint keeps idempotent)
        try:
            s.add(
                models.OrderTransition(
                    cid=cid,
                    transition_id=transition_id,
                    from_state=from_state,
                    to_state=to_state,
                    created_at=now,
                )
            )
            s.flush()
        except IntegrityError:
            s.rollback()
            # Ensure order update isn't lost: reopen transaction expectation is caller-managed.
            # For single-writer usage, IntegrityError only happens on duplicate transition_id insert.
            pass

        repo.system_events.write_event(
            event_type="ORDER_STATE_TRANSITION",
            correlation_id=cid,
            severity="INFO",
            symbol=order.symbol,
            payload={"cid": cid, "from": from_state, "to": to_state, "transition_id": transition_id},
        )

    def submit(self, s: Session, cid: str) -> SubmitResult:
        repo = Repo(s)
        order = s.get(models.Order, cid)
        if order is None:
            raise ValueError("Order not found")

        self.transition(s, cid, "CREATED", "SUBMITTED")

        dedupe_key = f"SEND_ORDER:{cid}"
        repo.outbox.enqueue(
            event_type="SEND_ORDER",
            dedupe_key=dedupe_key,
            payload={"cid": cid},
        )
        repo.system_events.write_event(
            event_type="OUTBOX_ENQUEUED",
            correlation_id=cid,
            severity="INFO",
            symbol=order.symbol,
            payload={"event_type": "SEND_ORDER", "dedupe_key": dedupe_key},
        )
        return SubmitResult(cid=cid, client_order_id=order.client_order_id, state="SUBMITTED")
