from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


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

    def send_order(self, raw_req: dict[str, Any]) -> BrokerSendResult:
        cid = str(raw_req["cid"])
        broker_order_id = f"B-{self._account_id}-{cid[:12]}"
        raw_response = {"ok": True, "broker_order_id": broker_order_id, "echo": raw_req.get("request_uuid")}
        return BrokerSendResult(broker_order_id=broker_order_id, raw_response=raw_response)

    def query_orders(self) -> list[dict[str, Any]]:
        return []

    def query_fills(self) -> list[dict[str, Any]]:
        return []
