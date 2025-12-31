from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.orm import Session

from app.config import settings
from app.adapters.ths_adapter import get_ths_adapter
from app.core.reasoning import ReasoningEngine
from app.core.guard import Guard
from app.core.order_manager import OrderManager
from app.core.reconciler import Reconciler
from app.core.source_fidelity import SourceFidelityEngine
from app.core.differential_audit import DifferentialAuditEngine
from app.core.outbox import OutboxDispatcher, BrokerSendResult
from app.database.engine import SessionLocal
from app.database.repo import Repo
from app.database import models
from app.utils.time import now_shanghai, trading_day_str, fmt_ts_millis
from app.utils.ids import make_cid
from app.utils.crypto import sha256_hex
from app.utils.p2 import P2Quantile


class MockBrokerSender:
    def send_order(self, raw_req: dict) -> BrokerSendResult:
        cid = raw_req["cid"]
        broker_order_id = f"B-{cid[:12]}"
        raw_response = {"ok": True, "broker_order_id": broker_order_id, "echo": raw_req["request_uuid"]}
        return BrokerSendResult(broker_order_id=broker_order_id, raw_response=raw_response)


@dataclass
class Orchestrator:
    running: bool = True

    def __post_init__(self) -> None:
        self.adapter = get_ths_adapter()
        self.reasoning = ReasoningEngine()
        self.guard = Guard()
        self.order_manager = OrderManager()
        self.reconciler = Reconciler()
        self.fidelity = SourceFidelityEngine()
        self.diff_audit = DifferentialAuditEngine()

        self.symbols = [x.strip() for x in settings.ORCH_SYMBOLS.split(",") if x.strip()]
        self.outbox_dispatcher = OutboxDispatcher(broker_sender=MockBrokerSender())

        self._last_tick_ts = None

    def stop(self) -> None:
        self.running = False

    async def run(self) -> None:
        while self.running:
            await self._tick()
            await asyncio.sleep(settings.ORCH_LOOP_INTERVAL_MS / 1000.0)

    def _trade_gate_ok(self, repo: Repo) -> bool:
        st = repo.system_status.get_for_update()

        if st.veto or st.guard_level >= 2 or st.panic_halt:
            return False

        if not settings.REQUIRE_SELF_CHECK_FOR_TRADING:
            return True

        if not st.last_self_check_report_hash or not st.last_self_check_time:
            return False

        age = (now_shanghai() - st.last_self_check_time).total_seconds()
        return age <= float(settings.SELF_CHECK_MAX_AGE_SEC)

    async def _tick(self) -> None:
        tick_now = now_shanghai()

        # drift monitoring
        with SessionLocal() as s0:
            repo0 = Repo(s0)
            if self._last_tick_ts is not None:
                real_ms = int(round((tick_now - self._last_tick_ts).total_seconds() * 1000))
                if abs(real_ms - int(settings.ORCH_LOOP_INTERVAL_MS)) >= int(settings.SCHED_DRIFT_THRESHOLD_MS):
                    repo0.system_events.write_event(
                        event_type="SCHEDULER_DRIFT",
                        correlation_id=None,
                        severity="WARN",
                        payload={"expected_ms": settings.ORCH_LOOP_INTERVAL_MS, "real_ms": real_ms},
                    )
                    s0.commit()
            self._last_tick_ts = tick_now

        # 1) ingest/decision/order enqueue
        with SessionLocal() as s:
            repo = Repo(s)

            # 9.1 Frozen daily versions
            try:
                repo.frozen_versions.ensure_today_frozen(
                    rule_set_version_hash=settings.RULESET_VERSION_HASH,
                    strategy_contract_hash=settings.STRATEGY_CONTRACT_HASH,
                    model_snapshot_uuid=settings.MODEL_SNAPSHOT_UUID,
                    cost_model_version=settings.COST_MODEL_VERSION,
                    canonicalization_version=settings.CANONICALIZATION_VERSION,
                    feature_extractor_version=settings.FEATURE_EXTRACTOR_VERSION,
                )
            except Exception as e:
                repo.system_events.write_event(
                    event_type="DAILY_FROZEN_VERSIONS_MISMATCH",
                    correlation_id=None,
                    severity="CRITICAL",
                    payload={"error": str(e)},
                )
                repo.system_status.set_panic_halt("FROZEN_VERSIONS_MISMATCH")
                s.commit()
                return

            for sym in self.symbols:
                if repo.symbol_lock.is_locked(sym):
                    repo.system_events.write_event(
                        event_type="SYMBOL_LOCKED_SKIP",
                        correlation_id=None,
                        severity="WARN",
                        symbol=sym,
                        payload={"reason": "symbol_locked"},
                    )
                    continue

                ev = self.adapter.fetch_market_event(sym)
                raw_row = self._ingest_event_with_sequencer(s, ev)

                features = {"price": float(ev["payload"].get("price", 0.0))}
                out = self.reasoning.infer(features)

                # decision bundle id deterministic
                decision_id = sha256_hex(f"{sym}|{raw_row.payload_sha256}|{out.reason_code}".encode("utf-8"))
                s.add(
                    models.DecisionBundle(
                        decision_id=decision_id,
                        cid=None,
                        decision=out.decision,
                        reason_code=out.reason_code,
                        params=out.params,
                        guard_status={},
                        data_quality={
                            "data_status": raw_row.data_status,
                            "latency_ms": raw_row.latency_ms,
                            "completion_rate": raw_row.completion_rate,
                            "realtime_flag": raw_row.realtime_flag,
                            "audit_flag": raw_row.audit_flag,
                        },
                        rule_set_version_hash=settings.RULESET_VERSION_HASH,
                        model_snapshot_uuid=settings.MODEL_SNAPSHOT_UUID,
                        strategy_contract_hash=settings.STRATEGY_CONTRACT_HASH,
                        feature_extractor_version=settings.FEATURE_EXTRACTOR_VERSION,
                        cost_model_version=settings.COST_MODEL_VERSION,
                        lineage_ref=raw_row.payload_sha256,
                        created_at=now_shanghai(),
                    )
                )

                if raw_row.audit_flag and raw_row.realtime_flag and (not raw_row.research_only):
                    s.add(
                        models.TrainingFeatureRow(
                            symbol=sym,
                            data_ts=raw_row.data_ts,
                            ingest_ts=raw_row.ingest_ts,
                            audit_flag=True,
                            realtime_equivalent=True,
                            payload_sha256=raw_row.payload_sha256,
                            channel_id=raw_row.channel_id,
                            channel_seq=raw_row.channel_seq,
                            source_clock_quality=raw_row.source_clock_quality,
                            feature_extractor_version=settings.FEATURE_EXTRACTOR_VERSION,
                            rule_set_version_hash=settings.RULESET_VERSION_HASH,
                            strategy_contract_hash=settings.STRATEGY_CONTRACT_HASH,
                            features=features,
                            created_at=now_shanghai(),
                        )
                    )

                if out.decision in {"BUY", "SELL"}:
                    if not self._trade_gate_ok(repo):
                        repo.system_events.write_event(
                            event_type="TRADE_GATE_BLOCKED",
                            correlation_id=None,
                            severity="WARN",
                            symbol=sym,
                            payload={"reason": "guard_or_self_check"},
                        )
                        continue

                    # signal time is decision time input (deterministic)
                    signal_dt = now_shanghai()
                    td = trading_day_str(signal_dt)
                    signal_ts_millis = fmt_ts_millis(signal_dt)

                    nonce = repo.nonce.next_nonce(sym, "PRIMARY")
                    cid = make_cid(
                        trading_day=td,
                        symbol=sym,
                        strategy_id="PRIMARY",
                        signal_ts_millis=signal_ts_millis,
                        nonce=nonce,
                        side=out.decision,
                        intended_qty_or_notional=100,
                    )

                    s.add(
                        models.Signal(
                            cid=cid,
                            trading_day=td,
                            symbol=sym,
                            strategy_id="PRIMARY",
                            signal_ts=signal_dt,
                            nonce=nonce,
                            side=out.decision,
                            intended_qty_or_notional=100,
                            confidence=out.confidence,
                            rule_set_version_hash=settings.RULESET_VERSION_HASH,
                            strategy_contract_hash=settings.STRATEGY_CONTRACT_HASH,
                            model_snapshot_uuid=settings.MODEL_SNAPSHOT_UUID,
                            cost_model_version=settings.COST_MODEL_VERSION,
                            feature_extractor_version=settings.FEATURE_EXTRACTOR_VERSION,
                            lineage_ref=raw_row.payload_sha256,
                            created_at=now_shanghai(),
                        )
                    )

                    gr = self.guard.evaluate(s, sym, out.decision, intended_qty=100)
                    if gr.veto:
                        repo.system_events.write_event(
                            event_type="GUARD_VETO",
                            correlation_id=cid,
                            severity="WARN",
                            symbol=sym,
                            payload={"guard_level": gr.guard_level, "veto_code": gr.veto_code},
                        )
                        continue

                    self.order_manager.create_order(
                        s=s,
                        cid=cid,
                        symbol=sym,
                        side=out.decision,
                        order_type="LIMIT",
                        limit_price=float(raw_row.payload.get("price", 0.0)),
                        qty=100,
                        strategy_contract_hash=settings.STRATEGY_CONTRACT_HASH,
                    )
                    self.order_manager.submit(s, cid)

            s.commit()

        # 2) outbox dispatch
        with SessionLocal() as s2:
            sent = self.outbox_dispatcher.pump_once(s2, limit=50)
            if sent:
                s2.commit()
            else:
                s2.rollback()

        # 3) reconcile fills (mock)
        with SessionLocal() as s3:
            fills = self.adapter.query_fills()
            if fills:
                self.reconciler.upsert_fills_first(s3, fills)
                s3.commit()

    def _ingest_event_with_sequencer(self, s: Session, ev: dict) -> models.RawMarketEvent:
        repo = Repo(s)

        data_ts = ev["data_ts"]
        ingest_ts = ev["ingest_ts"]
        latency_ms = int(round((ingest_ts - data_ts).total_seconds() * 1000))
        ev["latency_ms"] = max(0, latency_ms)

        channel_id = str(ev["channel_id"])
        seq = int(ev["channel_seq"])

        cur = s.get(models.ChannelCursor, channel_id)
        if cur is None:
            q = P2Quantile(q=0.99)
            q.update(float(ev["latency_ms"]))
            cur = models.ChannelCursor(
                channel_id=channel_id,
                last_seq=seq,
                last_ingest_ts=ingest_ts,
                quality_score=1.0,
                p99_latency_ms=max(settings.EPSILON_MIN_MS, int(round(q.value()))),
                p99_state=q.to_state(),
                fidelity_score=1.0,
                fidelity_low_streak=0,
                updated_at=now_shanghai(),
            )
            s.add(cur)
            s.flush()

        jitter = (seq <= int(cur.last_seq)) or (ingest_ts < cur.last_ingest_ts)

        q = P2Quantile.from_state(cur.p99_state or {"q": 0.99})
        q.update(float(ev["latency_ms"]))
        cur.p99_state = q.to_state()
        cur.p99_latency_ms = max(settings.EPSILON_MIN_MS, int(round(q.value())))
        cur.updated_at = now_shanghai()

        epsilon_ms = max(int(cur.p99_latency_ms), int(settings.EPSILON_MIN_MS))
        realtime_flag = (ingest_ts <= data_ts + timedelta(milliseconds=epsilon_ms))

        row = models.RawMarketEvent(
            api_schema_version=str(ev["api_schema_version"]),
            source=str(ev["source"]),
            ths_product=str(ev["ths_product"]),
            ths_function=str(ev["ths_function"]),
            ths_indicator_set=str(ev["ths_indicator_set"]),
            ths_params_canonical=str(ev["ths_params_canonical"]),
            ths_errorcode=str(ev["ths_errorcode"]),
            ths_quota_context=str(ev.get("ths_quota_context", "")),
            source_clock_quality=str(ev["source_clock_quality"]),
            channel_id=channel_id,
            channel_seq=seq,
            symbol=str(ev["symbol"]),
            data_ts=data_ts,
            ingest_ts=ingest_ts,
            payload=ev["payload"],
            payload_sha256=str(ev["payload_sha256"]),
            data_status=str(ev["data_status"]),
            latency_ms=int(ev["latency_ms"]),
            completion_rate=float(ev["completion_rate"]),
            realtime_flag=bool(realtime_flag),
            audit_flag=True,
            research_only=bool(jitter),
            request_id=str(ev["request_id"]),
            producer_instance=str(ev["producer_instance"]),
        )
        s.add(row)

        if jitter:
            cur.quality_score = max(0.0, float(cur.quality_score) - 0.1)
            repo.system_events.write_event(
                event_type="DATA_DEGRADED",
                correlation_id=row.request_id,
                severity="WARN",
                symbol=row.symbol,
                payload={"reason": "Channel_Jitter", "channel_id": channel_id, "seq": seq, "last_seq": int(cur.last_seq), "epsilon_ms": epsilon_ms},
            )
        else:
            cur.last_seq = seq
            cur.last_ingest_ts = ingest_ts

        return row
