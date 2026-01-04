from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any
from datetime import datetime

from app.config import settings
from app.utils.time import now_shanghai
from app.utils.crypto import sha256_hex
from app.utils.ids import new_request_id


class THSAdapter(Protocol):
    def fetch_market_event(self, symbol: str) -> dict[str, Any]: ...
    def query_orders(self) -> list[dict[str, Any]]: ...
    def query_fills(self) -> list[dict[str, Any]]: ...

    def fetch_minute_close_int64(self, symbol: str, data_ts: datetime, source: str) -> tuple[str, int]:
        """
        Return (channel_id, close_int64) for given source (DATAFEED/IFIND_HTTP/IFIND_SDK).
        """


@dataclass
class MockTHSAdapter:
    producer_instance: str = "mock-1"

    def fetch_market_event(self, symbol: str) -> dict[str, Any]:
        ingest_ts = now_shanghai()
        data_ts = ingest_ts

        payload = {"symbol": symbol, "price": 10.0, "ts": ingest_ts.isoformat()}
        payload_bytes = str(payload).encode("utf-8")

        req_id = new_request_id(f"{settings.DATA_PROVIDER}|{symbol}|{ingest_ts.isoformat()}")

        indicator_set = "PRICE"
        params_canonical = f"symbol={symbol}"

        return {
            "api_schema_version": settings.API_SCHEMA_VERSION,
            "source": settings.DATA_PROVIDER,
            "ths_product": settings.DATA_PROVIDER,
            "ths_function": "MOCK_PRICE",
            "ths_indicator_set": indicator_set,
            "ths_params_canonical": params_canonical,
            "ths_errorcode": "0",
            "ths_quota_context": "mock",

            "source_clock_quality": "AGGREGATED",
            "channel_id": f"{settings.DATA_PROVIDER}:{symbol}",
            "channel_seq": int(ingest_ts.timestamp() * 1000),
            "symbol": symbol,

            "data_ts": data_ts,
            "ingest_ts": ingest_ts,

            "payload": payload,
            "payload_sha256": sha256_hex(payload_bytes),

            "data_status": "VALID",
            "latency_ms": 0,
            "completion_rate": 1.0,

            "request_id": req_id,
            "producer_instance": self.producer_instance,
        }

    def query_orders(self) -> list[dict[str, Any]]:
        return []

    def query_fills(self) -> list[dict[str, Any]]:
        return []

    def fetch_minute_close_int64(self, symbol: str, data_ts: datetime, source: str) -> tuple[str, int]:
        base_price = 10.0
        if source == "DATAFEED":
            p = base_price
        elif source == "IFIND_HTTP":
            p = base_price + 0.01
        elif source == "IFIND_SDK":
            p = base_price
        else:
            p = base_price

        close_int64 = int(round(p * 10000))
        channel_id = f"{source}:{symbol}"
        return channel_id, close_int64


def get_ths_adapter() -> THSAdapter:
    return MockTHSAdapter()
