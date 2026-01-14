from __future__ import annotations

import re

_SYMBOL_RE = re.compile(r"^[0-9]{6}(?:\.(SZ|SH))?$", re.IGNORECASE)


def normalize_symbol(symbol: str | None) -> str:
    """Normalize A-share symbol.

    - Accepts: "000001", "000001.SZ", "600000", "600000.SH"
    - Returns uppercase, and appends exchange suffix for 6xxxxxx -> .SH else .SZ when missing.
    - If the input doesn't match the basic pattern, returns the stripped upper string as-is.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return ""

    m = _SYMBOL_RE.match(s)
    if not m:
        return s

    if "." in s:
        return s

    # Default A-share routing: 6xxxxxx/9xxxxxx -> SH, others -> SZ
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"
