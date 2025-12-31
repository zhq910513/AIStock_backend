from __future__ import annotations

from dataclasses import dataclass
import json

from sqlalchemy.orm import Session

from app.database.repo import Repo
from app.database import models
from app.utils.time import now_shanghai
from app.utils.crypto import sha256_hex
from app.core.order_manager import OrderManager


@dataclass
class BrokerSendResult:
    broker_order_id: str
    raw_response: dict


class OutboxDispatcher:
    """
    Outbox dispatcher:
    - SEND_ORDER: 发送后写 Orders.broker_order_id 并落 OrderAnchor（多锚点）
    """
    def __init__(self, broker_sender) -> None:
        self._broker_sender = broker_sender  # send_order(payload)->BrokerSendResult
        self._om = OrderManager()

    def pump_once(self, s: Session, limit: int = 50) -> int:
        repo = Repo(s)
        pending = repo.outbox.fetch_pending(limit=limit)
        sent = 0

        for ev in pending:
            try:
                if ev.event_type == "SEND_ORDER":
                    payload = dict(ev.payload or {})
                    cid = payload["cid"]
                    order = s.get(models.Order, cid)
                    if order is None:
                        repo.outbox.mark_failed(ev, "order_not_found")
                        continue

                    # deterministic request_uuid for audit (stable within this outbox row)
                    request_uuid = sha256_hex(f"SEND_ORDER|{cid}|{int(ev.id)}|{int(ev.attempts)}".encode("utf-8"))[:32]

                    raw_req = {
                        "request_uuid": request_uuid,
                        "client_order_id": order.client_order_id,
                        "cid": cid,
                        "symbol": order.symbol,
                        "side": order.side,
                        "order_type": order.order_type,
                        "limit_price_int64": int(order.limit_price_int64),
                        "qty_int": int(order.qty_int),
                        "metadata_hash": order.metadata_hash,
                    }
                    raw_request_hash = sha256_hex(json.dumps(raw_req, sort_keys=True).encode("utf-8"))

                    res = self._broker_sender.send_order(raw_req)  # includes raw_response
                    broker_order_id = res.broker_order_id
                    raw_resp = res.raw_response

                    raw_response_hash = sha256_hex(json.dumps(raw_resp, sort_keys=True).encode("utf-8"))
                    ack_hash = sha256_hex(f"{request_uuid}|{broker_order_id}|{raw_response_hash}".encode("utf-8"))

                    # Apply broker_order_id (mutable, OK)
                    order.broker_order_id = broker_order_id
                    order.updated_at = now_shanghai()

                    # State advance must go through state machine (idempotent)
                    # SUBMITTED -> PENDING
                    if order.state == "CREATED":
                        self._om.transition(s, cid, "CREATED", "SUBMITTED", transition_id=sha256_hex(f"AUTO_SUBMIT|{cid}".encode("utf-8"))[:32])
                        self._om.transition(s, cid, "SUBMITTED", "PENDING", transition_id=sha256_hex(f"BROKER_ACK|{cid}|{ack_hash}".encode("utf-8"))[:32])
                    elif order.state == "SUBMITTED":
                        self._om.transition(s, cid, "SUBMITTED", "PENDING", transition_id=sha256_hex(f"BROKER_ACK|{cid}|{ack_hash}".encode("utf-8"))[:32])

                    # anchor nail
                    repo.anchors.upsert_anchor(
                        cid=cid,
                        client_order_id=order.client_order_id,
                        broker_order_id=broker_order_id,
                        request_uuid=request_uuid,
                        ack_hash=ack_hash,
                        raw_request_hash=raw_request_hash,
                        raw_response_hash=raw_response_hash,
                    )

                    repo.system_events.write_event(
                        event_type="BROKER_ACK",
                        correlation_id=cid,
                        severity="INFO",
                        symbol=order.symbol,
                        payload={
                            "request_uuid": request_uuid,
                            "broker_order_id": broker_order_id,
                            "ack_hash": ack_hash,
                            "raw_request_hash": raw_request_hash,
                            "raw_response_hash": raw_response_hash,
                        },
                    )

                repo.outbox.mark_sent(ev)
                sent += 1
            except Exception as e:
                # SEND_ORDER is write op: do NOT auto retry to avoid duplicates.
                repo.outbox.mark_failed(ev, str(e))

        return sent
