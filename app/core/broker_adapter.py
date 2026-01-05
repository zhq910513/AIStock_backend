from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

from app.utils.crypto import sha256_hex
from app.utils.time import now_shanghai


@dataclass
class BrokerSendResult:
    broker_order_id: str
    raw_response: dict[str, Any]


class BrokerAdapter(Protocol):
    def send_order(self, raw_req: dict[str, Any]) -> BrokerSendResult: ...
    def query_orders(self) -> list[dict[str, Any]]: ...
    def query_fills(self) -> list[dict[str, Any]]: ...


class MockBrokerAdapter:
    """
    Deterministic mock broker.
    """
    def __init__(self, account_id: str) -> None:
        self._account_id = account_id
        self._orders: dict[str, dict[str, Any]] = {}
        self._fills: list[dict[str, Any]] = []

    def send_order(self, raw_req: dict[str, Any]) -> BrokerSendResult:
        cid = str(raw_req["cid"])
        broker_order_id = f"B-{self._account_id}-{cid[:12]}"
        raw_response = {"ok": True, "broker_order_id": broker_order_id, "echo": raw_req.get("request_uuid")}

        # record order (in-memory mock)
        self._orders[broker_order_id] = {
            "account_id": self._account_id,
            "broker_order_id": broker_order_id,
            "cid": cid,
            "symbol": str(raw_req.get("symbol", "")),
            "side": str(raw_req.get("side", "")),
            "order_type": str(raw_req.get("order_type", "")),
            "limit_price_int64": int(raw_req.get("limit_price_int64", 0)),
            "qty_int": int(raw_req.get("qty_int", 0)),
            "request_uuid": str(raw_req.get("request_uuid", "")),
            "created_at": now_shanghai().isoformat(),
        }

        # deterministic immediate fill for smoke testing
        fill_ts = now_shanghai()
        broker_fill_id = sha256_hex(
            f"FILL|{broker_order_id}|{raw_req.get('request_uuid','')}".encode("utf-8")
        )[:40]
        fill = {
            "broker_fill_id": broker_fill_id,
            "broker_order_id": broker_order_id,
            "cid": cid,
            "account_id": self._account_id,
            "symbol": str(raw_req.get("symbol", "")),
            "side": str(raw_req.get("side", "")),
            "fill_price_int64": int(raw_req.get("limit_price_int64", 0) or 0),
            "fill_qty_int": int(raw_req.get("qty_int", 0)),
            "fill_ts": fill_ts,
        }
        self._fills.append(fill)
        return BrokerSendResult(broker_order_id=broker_order_id, raw_response=raw_response)

    def query_orders(self) -> list[dict[str, Any]]:
        return list(self._orders.values())

    def query_fills(self) -> list[dict[str, Any]]:
        # drain-on-read to keep idempotency easier at caller
        out = list(self._fills)
        self._fills.clear()
        return out
