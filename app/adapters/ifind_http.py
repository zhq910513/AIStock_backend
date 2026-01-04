from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import json

import httpx

from app.config import settings
from app.utils.crypto import sha256_hex


@dataclass
class IFindHTTPResponse:
    http_status: int | None
    errorcode: str
    errmsg: str
    quota_context: str
    raw: dict[str, Any]
    payload_sha256: str


class IFindHTTPProvider:
    """
    iFind QuantAPI HTTP provider.

    Endpoints follow /api/v1/{endpoint}.

    Auth (per official examples): access_token is passed via HTTP header.
    """
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, timeout_sec: float = 8.0) -> None:
        self._base_url = (base_url or settings.IFIND_HTTP_BASE_URL).rstrip("/")
        self._token = token or settings.IFIND_HTTP_TOKEN
        self._timeout = timeout_sec
        self._client = httpx.Client(timeout=self._timeout)

    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse:
        url = f"{self._base_url}/api/v1/{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self._token:
            # iFind official examples place access_token in request headers.
            headers["access_token"] = self._token

        http_status: int | None = None
        raw: dict[str, Any] = {}
        errorcode = "0"
        errmsg = ""
        quota_context = ""

        try:
            r = self._client.post(url, headers=headers, json=payload)
            http_status = int(r.status_code)
            raw = r.json() if r.content else {}

            if isinstance(raw, dict):
                errorcode = str(raw.get("errorcode", "0"))
                errmsg = str(raw.get("errmsg", ""))

                dataVol = raw.get("dataVol")
                if isinstance(dataVol, dict):
                    quota_context = json.dumps(
                        {k: dataVol.get(k) for k in ("quota", "useCount", "remainCount") if k in dataVol},
                        ensure_ascii=False,
                    )

        except Exception as e:
            errorcode = "HTTP_EXCEPTION"
            errmsg = str(e)
            raw = {"error": str(e)}

        payload_sha256 = sha256_hex(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return IFindHTTPResponse(
            http_status=http_status,
            errorcode=errorcode,
            errmsg=errmsg,
            quota_context=quota_context,
            raw=raw,
            payload_sha256=payload_sha256,
        )
