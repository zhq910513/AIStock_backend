from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any
import json

from app.config import settings
from app.utils.crypto import sha256_hex
from app.adapters.ifind_http import IFindHTTPProvider, IFindHTTPResponse


class DataProvider(Protocol):
    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse: ...


@dataclass
class MockDataProvider:
    """
    Offline/dev provider: returns deterministic response.
    """
    mode: str = "MOCK"

    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse:
        raw = {"errorcode": "0", "errmsg": "", "endpoint": endpoint, "payload": payload, "mode": self.mode}
        payload_sha256 = sha256_hex(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return IFindHTTPResponse(
            http_status=200,
            errorcode="0",
            errmsg="",
            quota_context="",
            raw=raw,
            payload_sha256=payload_sha256,
        )


def get_data_provider() -> DataProvider:
    if settings.DATA_PROVIDER.upper() == "IFIND_HTTP":
        return IFindHTTPProvider()
    return MockDataProvider()
