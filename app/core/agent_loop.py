from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import timedelta

from sqlalchemy.orm import Session

from app.config import settings
from app.database import models
from app.database.repo import Repo
from app.core.data_dispatcher import DataRequestDispatcher
from app.core.feature_extractor import FeatureExtractor
from app.core.reasoning import ReasoningEngine
from app.utils.time import now_shanghai
from app.utils.crypto import sha256_hex


@dataclass
class AgentDecision:
    decision: str  # BUY/SELL/HOLD
    confidence: float
    reason_code: str
    params: dict[str, Any]
    request_ids: list[str]
    feature_hash: str
    model_hash: str
    validation_id: str | None


class AgentLoop:
    """
    PLAN -> ACT(enqueue) -> OBSERVE(dispatch) -> VERIFY -> DECIDE

    本轮升级点：
    - VERIFY 使用“突破/动量/量能/波动/盘中确认”的结构化证据
    - BUY 候选时主动追加盘中分钟数据请求（cmd_history_quotation + starttime/endtime） :contentReference[oaicite:1]{index=1}
    - validations 写入 evidence_checks 与 evidence_metrics，reason_code 可复盘
    """

    def __init__(self) -> None:
        self._dispatcher = DataRequestDispatcher()
        self._fx = FeatureExtractor()
        self._re = ReasoningEngine()

    def run_for_symbol(self, s: Session, symbol: str, account_id: str | None, correlation_id: str | None) -> AgentDecision:
        repo = Repo(s)
        req_ids: list[str] = []

        # --------------------
        # PLAN: realtime + daily history
        # --------------------
        rt_payload = {
            "codes": symbol,
            "indicators": "open,high,low,latest,close,volume,amount",
        }

        today = now_shanghai().date()
        startdate = (today - timedelta(days=60)).strftime("%Y-%m-%d")
        enddate = today.strftime("%Y-%m-%d")
        hist_payload = {
            "codes": symbol,
            "indicators": "open,high,low,close,volume,amount",
            "startdate": startdate,
            "enddate": enddate,
            "functionpara": {"Fill": "Blank"},
        }

        rid_rt = repo.data_requests.enqueue(
            dedupe_key=f"PLAN:RT:{symbol}:{now_shanghai().strftime('%Y%m%d%H%M')}",
            correlation_id=correlation_id,
            account_id=account_id,
            symbol=symbol,
            purpose="PLAN",
            provider=settings.DATA_PROVIDER,
            endpoint="real_time_quotation",
            params_canonical=f"codes={symbol}&indicators=open,high,low,latest,close,volume,amount",
            request_payload=rt_payload,
            deadline_sec=5,
        )
        req_ids.append(rid_rt)

        rid_hist = repo.data_requests.enqueue(
            dedupe_key=f"PLAN:HIST:{symbol}:{now_shanghai().strftime('%Y%m%d%H%M')}",
            correlation_id=correlation_id,
            account_id=account_id,
            symbol=symbol,
            purpose="PLAN",
            provider=settings.DATA_PROVIDER,
            endpoint="cmd_history_quotation",
            params_canonical=f"codes={symbol}&indicators=open,high,low,close,volume,amount&startdate={startdate}&enddate={enddate}",
            request_payload=hist_payload,
            deadline_sec=8,
        )
        req_ids.append(rid_hist)

        s.flush()
        self._dispatcher.pump_once(s, limit=settings.AGENT_MAX_REQUESTS_PER_SYMBOL)

        rt_raw = self._load_response_raw(s, rid_rt)
        hist_raw = self._load_response_raw(s, rid_hist)

        # --------------------
        # FEATURES + MODEL
        # --------------------
        fxr0 = self._fx.extract(symbol=symbol, realtime_raw=rt_raw, history_raw=hist_raw, intraday_raw=None)
        ro0 = self._re.infer(fxr0.features)

        # --------------------
        # VERIFY (Phase 1: daily evidence)
        # --------------------
        checks, metrics, phase1_pass = self._verify_daily(fxr0.features, ro0)

        intraday_raw = None
        rid_intra = None

        # BUY 候选才拉盘中分钟K（主动索取数据验证）
        buy_candidate = (ro0.confidence >= settings.AGENT_VERIFY_MIN_CONFIDENCE) and phase1_pass and (ro0.score > 0.01)

        if buy_candidate:
            # cmd_history_quotation 支持 starttime/endtime 用于盘中分钟/时点历史行情 :contentReference[oaicite:2]{index=2}
            starttime = f"{today.strftime('%Y-%m-%d')} 09:30:00"
            endtime = now_shanghai().strftime("%Y-%m-%d %H:%M:%S")

            intra_payload = {
                "codes": symbol,
                "indicators": "open,high,low,close,volume,amount",
                "starttime": starttime,
                "endtime": endtime,
            }

            rid_intra = repo.data_requests.enqueue(
                dedupe_key=f"VERIFY:INTRA:{symbol}:{now_shanghai().strftime('%Y%m%d%H%M%S')}",
                correlation_id=correlation_id,
                account_id=account_id,
                symbol=symbol,
                purpose="VERIFY",
                provider=settings.DATA_PROVIDER,
                endpoint="cmd_history_quotation",
                params_canonical=f"codes={symbol}&indicators=open,high,low,close,volume,amount&starttime={starttime}&endtime={endtime}",
                request_payload=intra_payload,
                deadline_sec=8,
            )
            req_ids.append(rid_intra)

            s.flush()
            self._dispatcher.pump_once(s, limit=settings.AGENT_MAX_REQUESTS_PER_SYMBOL)
            intraday_raw = self._load_response_raw(s, rid_intra)

        # 二次提特征（带盘中数据）
        fxr = self._fx.extract(symbol=symbol, realtime_raw=rt_raw, history_raw=hist_raw, intraday_raw=intraday_raw)
        ro = self._re.infer(fxr.features)

        # --------------------
        # VERIFY (Phase 2: intraday evidence if any)
        # --------------------
        phase2_pass = True
        if intraday_raw is not None:
            phase2_checks, phase2_metrics, phase2_pass = self._verify_intraday(fxr.features)
            checks.update(phase2_checks)
            metrics.update(phase2_metrics)

        # 总结 PASS/INCONCLUSIVE
        # 规则：Phase1 必须过；若拉了 Phase2，则 Phase2 也必须过
        passes = phase1_pass and phase2_pass and (ro.confidence >= settings.AGENT_VERIFY_MIN_CONFIDENCE)

        hypothesis = (
            f"Hold {settings.HOLD_DAYS_MIN}-{settings.HOLD_DAYS_MAX} days, "
            f"target return {settings.TARGET_RETURN_MIN:.2%}-{settings.TARGET_RETURN_MAX:.2%}"
        )

        conclusion = "PASS" if passes else "INCONCLUSIVE"
        evidence = {
            "checks": checks,
            "metrics": metrics,
            "model": {"score": float(ro.score), "confidence": float(ro.confidence), "reason_code": ro.reason_code},
            "features_hash": fxr.feature_hash,
            "request_ids": req_ids,
        }

        # validation_id
        val_id = repo.validations.write(
            decision_id="PENDING",
            symbol=symbol,
            hypothesis=hypothesis,
            request_ids=req_ids,
            evidence=evidence,
            conclusion=conclusion,
            score=float(ro.confidence),
        )

        # --------------------
        # DECIDE + reason_code mapping
        # --------------------
        decision = "HOLD"
        reason_code = "RC_AGENT_INCONCLUSIVE_V1"
        params = dict(ro.params or {})
        params.update(
            {
                "confidence": float(ro.confidence),
                "hypothesis": hypothesis,
                "validation_conclusion": conclusion,
                "validation_id": val_id,
                "evidence_checks": checks,
                "evidence_metrics": metrics,
            }
        )

        if conclusion == "PASS" and float(ro.score) > 0.01:
            decision = "BUY"
            reason_code = self._reason_code_for_buy(checks)
        elif float(ro.score) < -0.01:
            decision = "SELL"
            reason_code = "RC_AGENT_RISK_OFF_SELL_V1"

        return AgentDecision(
            decision=decision,
            confidence=float(ro.confidence),
            reason_code=reason_code,
            params=params,
            request_ids=req_ids,
            feature_hash=fxr.feature_hash,
            model_hash=self._re.model_hash(),
            validation_id=val_id,
        )

    # -------------------------
    # Verify rules (structured evidence)
    # -------------------------
    def _verify_daily(self, f: dict[str, float], ro: Any) -> tuple[dict[str, bool], dict[str, float], bool]:
        checks: dict[str, bool] = {}
        metrics: dict[str, float] = {}

        daily_n = int(f.get("daily_n", 0.0))
        metrics["daily_n"] = float(daily_n)

        # 基本数据充足性
        checks["daily_enough_bars"] = daily_n >= int(settings.AGENT_MIN_DAILY_BARS)

        # 突破：20日突破（close>prev high）
        breakout = f.get("breakout_20d", 0.0) >= 0.5
        checks["daily_breakout_20d"] = breakout

        # 3日动量（不要太弱）
        mom3 = float(f.get("momentum_3d", 0.0))
        metrics["momentum_3d"] = mom3
        checks["daily_momentum_3d_ok"] = mom3 >= float(settings.AGENT_MIN_MOMENTUM_3D)

        # 波动过滤（太大波动先不打）
        volp = float(f.get("vol_proxy", 1.0))
        metrics["vol_proxy"] = volp
        checks["daily_vol_ok"] = volp <= float(settings.AGENT_MAX_DAILY_VOL_PROXY)

        # 量能放大（突破更可信）
        vs = f.get("volume_surge", 0.0) >= 0.5
        checks["daily_volume_surge"] = vs

        # 收盘接近近期高位（趋势更强）
        c2h = float(f.get("close_to_high_5d", 0.0))
        metrics["close_to_high_5d"] = c2h
        checks["daily_close_near_high"] = c2h >= 0.97

        # 组合：为了贴近 1-3 天游走 5-8%
        # 先用 “突破 + 波动不过大” 作为硬门槛，动量/量能/收盘位置为加分项
        hard = checks["daily_enough_bars"] and checks["daily_breakout_20d"] and checks["daily_vol_ok"]
        soft_score = sum(
            [
                1 if checks["daily_momentum_3d_ok"] else 0,
                1 if checks["daily_volume_surge"] else 0,
                1 if checks["daily_close_near_high"] else 0,
            ]
        )
        metrics["daily_soft_score"] = float(soft_score)
        phase1_pass = hard and soft_score >= 1

        return checks, metrics, phase1_pass

    def _verify_intraday(self, f: dict[str, float]) -> tuple[dict[str, bool], dict[str, float], bool]:
        checks: dict[str, bool] = {}
        metrics: dict[str, float] = {}

        n = int(f.get("intraday_n", 0.0))
        metrics["intraday_n"] = float(n)
        checks["intraday_enough_bars"] = n >= int(settings.AGENT_INTRADAY_MIN_BARS)

        dd = float(f.get("intraday_max_dd", 0.0))  # negative
        metrics["intraday_max_dd"] = dd
        checks["intraday_dd_ok"] = dd >= -float(settings.AGENT_INTRADAY_MAX_DRAWDOWN)

        above_vwap = f.get("intraday_above_vwap", 0.0) >= 0.5
        metrics["intraday_above_vwap"] = 1.0 if above_vwap else 0.0
        checks["intraday_above_vwap"] = (not bool(settings.AGENT_INTRADAY_REQUIRE_ABOVE_VWAP)) or above_vwap

        c2h = float(f.get("intraday_close_to_high", 0.0))
        metrics["intraday_close_to_high"] = c2h
        checks["intraday_close_near_high"] = c2h >= 0.985

        # 盘中确认：不允许明显走弱（深回撤），最好站上 VWAP 且收盘靠近日高
        hard = checks["intraday_enough_bars"] and checks["intraday_dd_ok"] and checks["intraday_above_vwap"]
        soft = 1 if checks["intraday_close_near_high"] else 0
        metrics["intraday_soft_score"] = float(soft)

        return checks, metrics, (hard and soft >= 0)

    def _reason_code_for_buy(self, checks: dict[str, bool]) -> str:
        # 让 reason_code 一眼看懂“为什么买”
        b = checks.get("daily_breakout_20d", False)
        v = checks.get("daily_volume_surge", False)
        m = checks.get("daily_momentum_3d_ok", False)
        i = checks.get("intraday_above_vwap", False)
        d = checks.get("intraday_dd_ok", False)

        if b and v and i and d:
            return "RC_SWING_BREAKOUT_VOL_VWAP_DD_OK_V1"
        if b and v and m:
            return "RC_SWING_BREAKOUT_VOL_MOM_V1"
        if b and i:
            return "RC_SWING_BREAKOUT_INTRADAY_CONFIRM_V1"
        return "RC_SWING_BASE_PASS_BUY_V1"

    # -------------------------
    # DB response loader
    # -------------------------
    def _load_response_raw(self, s: Session, request_id: str) -> dict[str, Any] | None:
        req = s.get(models.DataRequest, request_id)
        if req is None or not req.response_id:
            return None
        resp = s.get(models.DataResponse, req.response_id)
        if resp is None:
            return None
        return dict(resp.raw or {})
