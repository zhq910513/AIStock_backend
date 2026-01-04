from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import math
from statistics import mean, pstdev

from app.config import settings
from app.utils.crypto import sha256_hex


@dataclass
class FeatureExtractionResult:
    features: dict[str, float]
    feature_hash: str


class FeatureExtractor:
    """
    将 iFind HTTP 返回(raw JSON)解析成稳定的最小特征集。
    - 优先解析 raw["tables"][0]["table"] 结构（常见 HTTP 返回）
    - 不依赖具体字段名以外的复杂结构；解析失败时给保守默认值
    """

    def extract(
        self,
        symbol: str,
        realtime_raw: dict[str, Any] | None,
        history_raw: dict[str, Any] | None,
        intraday_raw: dict[str, Any] | None = None,
    ) -> FeatureExtractionResult:
        feats: dict[str, float] = {}

        # -------- realtime --------
        # realtime 主要用 latest 作为补充（历史也会有 close）
        rt_latest = self._safe_latest_price(realtime_raw)
        if rt_latest is not None:
            feats["rt_latest"] = float(rt_latest)

        # -------- daily history --------
        daily = self._parse_ifind_table(history_raw)
        daily_feats = self._daily_features(daily)
        feats.update(daily_feats)

        # -------- intraday minute --------
        intr = self._parse_ifind_table(intraday_raw)
        intr_feats = self._intraday_features(intr)
        feats.update(intr_feats)

        # -------- stable hash --------
        feature_hash = sha256_hex(json.dumps(feats, sort_keys=True).encode("utf-8"))
        return FeatureExtractionResult(features=feats, feature_hash=feature_hash)

    # -------------------------
    # Parsing helpers
    # -------------------------
    def _safe_latest_price(self, raw: dict[str, Any] | None) -> float | None:
        if not raw or not isinstance(raw, dict):
            return None
        # common places: tables[0].table.latest[0]
        tbl = self._parse_ifind_table(raw)
        if not tbl:
            return None
        latest = self._first_number(tbl.get("latest"))
        if latest is None:
            latest = self._first_number(tbl.get("close"))
        return float(latest) if latest is not None else None

    def _parse_ifind_table(self, raw: dict[str, Any] | None) -> dict[str, Any] | None:
        if not raw or not isinstance(raw, dict):
            return None

        tables = raw.get("tables")
        # sometimes tables is dict, sometimes list
        if isinstance(tables, list) and tables:
            t0 = tables[0]
            if isinstance(t0, dict):
                tbl = t0.get("table")
                return tbl if isinstance(tbl, dict) else None
        if isinstance(tables, dict):
            # fallback: tables might directly contain "table"
            tbl = tables.get("table")
            return tbl if isinstance(tbl, dict) else None
        return None

    def _first_number(self, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list) and v:
            x = v[0]
            if isinstance(x, (int, float)):
                return float(x)
            try:
                return float(x)
            except Exception:
                return None
        try:
            return float(v)
        except Exception:
            return None

    def _series(self, tbl: dict[str, Any] | None, key: str) -> list[float]:
        if not tbl or not isinstance(tbl, dict):
            return []
        v = tbl.get(key)
        if v is None:
            return []
        if isinstance(v, list):
            out: list[float] = []
            for x in v:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out
        # sometimes single value
        try:
            return [float(v)]
        except Exception:
            return []

    # -------------------------
    # Feature builders
    # -------------------------
    def _daily_features(self, tbl: dict[str, Any] | None) -> dict[str, float]:
        out: dict[str, float] = {}

        close = self._series(tbl, "close")
        high = self._series(tbl, "high")
        low = self._series(tbl, "low")
        vol = self._series(tbl, "volume")

        n = len(close)
        out["daily_n"] = float(n)

        if n < 2:
            # 极度保守默认
            out["ret_1d"] = 0.0
            out["momentum_3d"] = 0.0
            out["vol_proxy"] = 1.0
            out["breakout_20d"] = 0.0
            out["volume_surge"] = 0.0
            out["close_to_high_5d"] = 0.0
            return out

        # returns
        out["ret_1d"] = (close[-1] / close[-2] - 1.0) if close[-2] != 0 else 0.0
        if n >= 4 and close[-4] != 0:
            out["momentum_3d"] = (close[-1] / close[-4] - 1.0)
        else:
            out["momentum_3d"] = 0.0

        # vol proxy: std of daily returns (use up to last 20 bars)
        rets: list[float] = []
        for i in range(1, n):
            if close[i - 1] != 0:
                rets.append(close[i] / close[i - 1] - 1.0)
        window = rets[-20:] if len(rets) >= 20 else rets
        out["vol_proxy"] = float(pstdev(window)) if len(window) >= 2 else 0.0

        # breakout: close above max(high) of lookback (excluding today)
        look = int(settings.AGENT_BREAKOUT_LOOKBACK_DAYS)
        if look < 5:
            look = 5
        if len(high) >= look + 1:
            prev_high = max(high[-(look + 1) : -1])
            out["breakout_20d"] = 1.0 if close[-1] > prev_high else 0.0
        else:
            out["breakout_20d"] = 0.0

        # volume surge vs prior mean
        if len(vol) >= 6:
            prior = vol[:-1]
            base = mean(prior[-20:]) if len(prior) >= 20 else mean(prior)
            mult = float(settings.AGENT_VOLUME_SURGE_MULT)
            out["volume_surge"] = 1.0 if (base > 0 and vol[-1] >= base * mult) else 0.0
        else:
            out["volume_surge"] = 0.0

        # close near recent highs: close / max(high_5d)
        if len(high) >= 5:
            h5 = max(high[-5:])
            out["close_to_high_5d"] = float(close[-1] / h5) if h5 > 0 else 0.0
        else:
            out["close_to_high_5d"] = 0.0

        return out

    def _intraday_features(self, tbl: dict[str, Any] | None) -> dict[str, float]:
        out: dict[str, float] = {}

        close = self._series(tbl, "close")
        open_ = self._series(tbl, "open")
        high = self._series(tbl, "high")
        low = self._series(tbl, "low")
        vol = self._series(tbl, "volume")

        n = len(close)
        out["intraday_n"] = float(n)
        if n == 0:
            out["intraday_ret"] = 0.0
            out["intraday_max_dd"] = 0.0
            out["intraday_above_vwap"] = 0.0
            out["intraday_close_to_high"] = 0.0
            out["intraday_vol_sum"] = 0.0
            return out

        # intraday return: last close / first open
        o0 = open_[0] if open_ else close[0]
        out["intraday_ret"] = (close[-1] / o0 - 1.0) if o0 != 0 else 0.0

        # max drawdown from intraday peak close (negative number)
        peak = -math.inf
        max_dd = 0.0
        for c in close:
            if c > peak:
                peak = c
            if peak > 0:
                dd = c / peak - 1.0
                if dd < max_dd:
                    max_dd = dd
        out["intraday_max_dd"] = float(max_dd)

        # VWAP: sum(price*vol)/sum(vol)
        v_sum = sum(vol) if vol else 0.0
        if v_sum > 0 and len(vol) == len(close):
            vwap = sum(close[i] * vol[i] for i in range(n)) / v_sum
            out["intraday_above_vwap"] = 1.0 if close[-1] >= vwap else 0.0
        else:
            out["intraday_above_vwap"] = 0.0

        # close to intraday high
        if high:
            h = max(high)
            out["intraday_close_to_high"] = float(close[-1] / h) if h > 0 else 0.0
        else:
            out["intraday_close_to_high"] = 0.0

        out["intraday_vol_sum"] = float(v_sum)
        return out
