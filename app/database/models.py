from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    BigInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class RawMarketEvent(Base):
    __tablename__ = "raw_market_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    api_schema_version = Column(String(32), nullable=False)

    source = Column(String(32), nullable=False)
    ths_product = Column(String(32), nullable=False)
    ths_function = Column(String(128), nullable=False)
    ths_indicator_set = Column(String(512), nullable=False)
    ths_params_canonical = Column(String(2048), nullable=False)
    ths_errorcode = Column(String(64), nullable=False, default="0")
    ths_quota_context = Column(String(512), nullable=False, default="")

    source_clock_quality = Column(String(32), nullable=False)
    channel_id = Column(String(64), nullable=False)
    channel_seq = Column(BigInteger, nullable=False)
    symbol = Column(String(32), nullable=False)

    data_ts = Column(DateTime(timezone=True), nullable=False, index=True)
    ingest_ts = Column(DateTime(timezone=True), nullable=False, index=True)

    payload = Column(JSONB, nullable=False)
    payload_sha256 = Column(String(64), nullable=False)

    data_status = Column(String(16), nullable=False, default="VALID")
    latency_ms = Column(Integer, nullable=False, default=0)
    completion_rate = Column(Float, nullable=False, default=1.0)

    realtime_flag = Column(Boolean, nullable=False, default=True)
    audit_flag = Column(Boolean, nullable=False, default=True)
    research_only = Column(Boolean, nullable=False, default=False)

    request_id = Column(String(64), nullable=False)
    producer_instance = Column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("channel_id", "channel_seq", name="uq_event_channel_seq"),
        Index("ix_event_symbol_data_ts", "symbol", "data_ts"),
    )


class ChannelCursor(Base):
    __tablename__ = "channel_cursor"

    channel_id = Column(String(64), primary_key=True)
    last_seq = Column(BigInteger, nullable=False, default=0)
    last_ingest_ts = Column(DateTime(timezone=True), nullable=False)
    quality_score = Column(Float, nullable=False, default=1.0)

    p99_latency_ms = Column(Integer, nullable=False, default=200)
    p99_state = Column(JSONB, nullable=False, default=dict)

    fidelity_score = Column(Float, nullable=False, default=1.0)
    fidelity_low_streak = Column(Integer, nullable=False, default=0)

    updated_at = Column(DateTime(timezone=True), nullable=False)


class InstrumentRuleCache(Base):
    __tablename__ = "instrument_rule_cache"

    symbol = Column(String(32), primary_key=True)
    tick_rule_version = Column(String(64), nullable=False)
    lot_rule_version = Column(String(64), nullable=False)
    tick_size = Column(Float, nullable=False)
    lot_size = Column(Integer, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class NonceCursor(Base):
    __tablename__ = "nonce_cursor"

    trading_day = Column(String(8), primary_key=True)
    symbol = Column(String(32), primary_key=True)
    strategy_id = Column(String(64), primary_key=True)
    last_nonce = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class SymbolLock(Base):
    __tablename__ = "symbol_locks"

    symbol = Column(String(32), primary_key=True)
    locked = Column(Boolean, nullable=False, default=False)
    lock_reason = Column(String(64), nullable=False, default="")
    lock_ref = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class Signal(Base):
    __tablename__ = "signals"

    cid = Column(String(64), primary_key=True)
    trading_day = Column(String(8), nullable=False)
    symbol = Column(String(32), nullable=False)
    strategy_id = Column(String(64), nullable=False)
    signal_ts = Column(DateTime(timezone=True), nullable=False)

    nonce = Column(Integer, nullable=False)
    side = Column(String(8), nullable=False)
    intended_qty_or_notional = Column(BigInteger, nullable=False)

    confidence = Column(Float, nullable=False, default=0.0)

    rule_set_version_hash = Column(String(64), nullable=False)
    strategy_contract_hash = Column(String(64), nullable=False)
    model_snapshot_uuid = Column(String(64), nullable=False)
    cost_model_version = Column(String(64), nullable=False)
    feature_extractor_version = Column(String(64), nullable=False)

    lineage_ref = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_signals_symbol_day", "symbol", "trading_day"),)


class DecisionBundle(Base):
    __tablename__ = "decision_bundles"

    decision_id = Column(String(64), primary_key=True)
    cid = Column(String(64), nullable=True, index=True)

    decision = Column(String(8), nullable=False)
    reason_code = Column(String(64), nullable=False)
    params = Column(JSONB, nullable=False, default=dict)

    guard_status = Column(JSONB, nullable=False, default=dict)
    data_quality = Column(JSONB, nullable=False, default=dict)

    rule_set_version_hash = Column(String(64), nullable=False)
    model_snapshot_uuid = Column(String(64), nullable=False)
    strategy_contract_hash = Column(String(64), nullable=False)
    feature_extractor_version = Column(String(64), nullable=False)
    cost_model_version = Column(String(64), nullable=False)

    lineage_ref = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class RuleSet(Base):
    __tablename__ = "rule_sets"
    rule_set_version_hash = Column(String(64), primary_key=True)
    definition = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class StrategyContract(Base):
    __tablename__ = "strategy_contracts"
    strategy_contract_hash = Column(String(64), primary_key=True)
    definition = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class DailyFrozenVersions(Base):
    __tablename__ = "daily_frozen_versions"

    trading_day = Column(String(8), primary_key=True)
    rule_set_version_hash = Column(String(64), nullable=False)
    strategy_contract_hash = Column(String(64), nullable=False)
    model_snapshot_uuid = Column(String(64), nullable=False)
    cost_model_version = Column(String(64), nullable=False)
    canonicalization_version = Column(String(32), nullable=False)
    feature_extractor_version = Column(String(64), nullable=False)

    report_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class Order(Base):
    __tablename__ = "orders"

    cid = Column(String(64), primary_key=True)
    client_order_id = Column(String(80), nullable=False, unique=True)
    broker_order_id = Column(String(80), nullable=True, index=True)

    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    order_type = Column(String(16), nullable=False)

    limit_price_int64 = Column(BigInteger, nullable=False, default=0)
    qty_int = Column(BigInteger, nullable=False)

    tick_rule_version = Column(String(64), nullable=False)
    lot_rule_version = Column(String(64), nullable=False)
    canonicalization_version = Column(String(32), nullable=False)

    metadata_hash = Column(String(64), nullable=False, index=True)

    state = Column(String(20), nullable=False, default="CREATED")
    version_id = Column(Integer, nullable=False, default=1)

    # Spec-required: last_transition_id gate
    last_transition_id = Column(String(64), nullable=True)

    strategy_contract_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint("qty_int >= 0", name="ck_order_qty_nonneg"),
        Index("ix_orders_state", "state"),
    )


class OrderTransition(Base):
    __tablename__ = "order_transitions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    cid = Column(String(64), ForeignKey("orders.cid", ondelete="CASCADE"), nullable=False, index=True)
    transition_id = Column(String(64), nullable=False)
    from_state = Column(String(20), nullable=False)
    to_state = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (UniqueConstraint("cid", "transition_id", name="uq_order_transition_cid_tid"),)
    order = relationship("Order")


class OrderAnchor(Base):
    __tablename__ = "order_anchors"

    cid = Column(String(64), primary_key=True)
    client_order_id = Column(String(80), nullable=False, index=True)
    broker_order_id = Column(String(80), nullable=True, index=True)

    request_uuid = Column(String(64), nullable=False, unique=True)
    ack_hash = Column(String(64), nullable=False)
    raw_request_hash = Column(String(64), nullable=False)
    raw_response_hash = Column(String(64), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_order_anchor_broker", "broker_order_id"),
    )


class TradeFill(Base):
    __tablename__ = "trade_fills"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    broker_fill_id = Column(String(128), nullable=False, unique=True)

    cid = Column(String(64), nullable=True, index=True)
    broker_order_id = Column(String(80), nullable=True, index=True)

    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)

    fill_price_int64 = Column(BigInteger, nullable=False)
    fill_qty_int = Column(BigInteger, nullable=False)
    fill_ts = Column(DateTime(timezone=True), nullable=False, index=True)

    fill_fingerprint = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False)


class TradeFillLink(Base):
    """
    Mutable mapping layer to avoid updating immutable Trade_Fills.
    Link is created by signed reconcile decision / compensation event.
    """
    __tablename__ = "trade_fill_links"

    fill_fingerprint = Column(String(64), primary_key=True)
    cid = Column(String(64), nullable=False, index=True)
    broker_order_id = Column(String(80), nullable=True)

    snapshot_id = Column(String(64), nullable=True, index=True)
    decision_id = Column(String(64), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), nullable=False)


class ReconcileSnapshot(Base):
    __tablename__ = "reconcile_snapshots"
    snapshot_id = Column(String(64), primary_key=True)

    # These are required base columns (0001 creates them; later migrations may add indexes)
    symbol = Column(String(32), nullable=False, index=True)
    anchor_type = Column(String(32), nullable=False)
    anchor_fingerprint = Column(String(64), nullable=False, index=True)

    candidates = Column(JSONB, nullable=False)
    report_hash = Column(String(64), nullable=False)

    status = Column(String(32), nullable=False, default="OPEN")
    created_at = Column(DateTime(timezone=True), nullable=False)


class ReconcileDecision(Base):
    __tablename__ = "reconcile_decisions"
    decision_id = Column(String(64), primary_key=True)
    snapshot_id = Column(String(64), ForeignKey("reconcile_snapshots.snapshot_id"), nullable=False, index=True)

    decided_cid = Column(String(64), nullable=False)
    decided_broker_order_id = Column(String(80), nullable=True)

    signer_key_id = Column(String(64), nullable=False)
    signature = Column(String(512), nullable=False)

    prev_decision_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)

    snapshot = relationship("ReconcileSnapshot")


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False, index=True)
    dedupe_key = Column(String(128), nullable=False, unique=True)

    status = Column(String(16), nullable=False, default="PENDING")
    attempts = Column(Integer, nullable=False, default=0)

    payload = Column(JSONB, nullable=False, default=dict)
    last_error = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"
    symbol = Column(String(32), primary_key=True)
    current_qty = Column(BigInteger, nullable=False, default=0)
    frozen_qty = Column(BigInteger, nullable=False, default=0)
    avg_price_int64 = Column(BigInteger, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class TradeLog(Base):
    __tablename__ = "trade_log"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    correlation_id = Column(String(64), nullable=False, index=True)
    cid = Column(String(64), nullable=True, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    execution_state = Column(String(32), nullable=False)
    feature_snapshot = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False)


class T1Constraint(Base):
    __tablename__ = "t1_constraints"
    symbol = Column(String(32), primary_key=True)
    locked_qty = Column(BigInteger, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class TrainingFeatureRow(Base):
    __tablename__ = "training_feature_store"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)

    data_ts = Column(DateTime(timezone=True), nullable=False)
    ingest_ts = Column(DateTime(timezone=True), nullable=False)

    audit_flag = Column(Boolean, nullable=False)
    realtime_equivalent = Column(Boolean, nullable=False)

    payload_sha256 = Column(String(64), nullable=False)
    channel_id = Column(String(64), nullable=False)
    channel_seq = Column(BigInteger, nullable=False)
    source_clock_quality = Column(String(32), nullable=False)

    feature_extractor_version = Column(String(64), nullable=False)
    rule_set_version_hash = Column(String(64), nullable=False)
    strategy_contract_hash = Column(String(64), nullable=False)

    features = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint("audit_flag = true AND realtime_equivalent = true", name="ck_training_only_realtime_equiv"),
        Index("ix_training_symbol_ts", "symbol", "data_ts"),
    )


class ModelSnapshot(Base):
    __tablename__ = "model_snapshots"
    model_snapshot_uuid = Column(String(64), primary_key=True)
    weights = Column(JSONB, nullable=False, default=dict)
    eval_report = Column(JSONB, nullable=False, default=dict)
    cost_model_version = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class ShadowRun(Base):
    __tablename__ = "shadow_runs"
    run_id = Column(String(64), primary_key=True)
    mode = Column(String(32), nullable=False)
    old_contract_hash = Column(String(64), nullable=False)
    new_contract_hash = Column(String(64), nullable=False)
    summary = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False)


class SourceFidelityDaily(Base):
    __tablename__ = "source_fidelity_daily"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    data_ts = Column(DateTime(timezone=True), nullable=False, index=True)

    channel_id_a = Column(String(64), nullable=False)
    channel_id_b = Column(String(64), nullable=False)

    close_a = Column(BigInteger, nullable=False)
    close_b = Column(BigInteger, nullable=False)
    abs_diff = Column(BigInteger, nullable=False)
    threshold = Column(BigInteger, nullable=False)

    fidelity_score_before = Column(Float, nullable=False)
    fidelity_score_after = Column(Float, nullable=False)
    action_taken = Column(String(32), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_fidelity_symbol_ts", "symbol", "data_ts"),)


class GuardianKey(Base):
    __tablename__ = "guardian_keys"
    key_id = Column(String(64), primary_key=True)
    role = Column(String(32), nullable=False)  # DEVOPS / RISK_OFFICER / COMPLIANCE
    public_key_b64 = Column(String(512), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class SystemStatus(Base):
    __tablename__ = "system_status"
    id = Column(Integer, primary_key=True, default=1)

    guard_level = Column(Integer, nullable=False, default=0)
    veto = Column(Boolean, nullable=False, default=False)
    veto_code = Column(String(64), nullable=False, default="")
    panic_halt = Column(Boolean, nullable=False, default=False)
    challenge_code = Column(String(128), nullable=True)

    last_self_check_report_hash = Column(String(64), nullable=True)
    last_self_check_time = Column(DateTime(timezone=True), nullable=True)

    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (CheckConstraint("id = 1", name="ck_system_status_singleton"),)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False, default="INFO")

    correlation_id = Column(String(64), nullable=True, index=True)
    symbol = Column(String(32), nullable=True, index=True)

    payload = Column(JSONB, nullable=False, default=dict)
    time = Column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (Index("ix_system_events_type_time", "event_type", "time"),)
