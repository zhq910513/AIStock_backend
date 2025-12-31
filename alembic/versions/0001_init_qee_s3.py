"""init qee-s3 (aligned with models)

Revision ID: 0001_init_qee_s3
Revises:
Create Date: 2025-12-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_init_qee_s3"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------
    # system_status / system_events
    # ---------------------------
    op.create_table(
        "system_status",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("guard_level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("veto", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("veto_code", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("panic_halt", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("challenge_code", sa.String(length=128), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_system_status_singleton"),
    )

    op.create_table(
        "system_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False, server_default="INFO"),
        sa.Column("correlation_id", sa.String(length=64), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_system_events_type_time", "system_events", ["event_type", "time"], unique=False)
    op.create_index("ix_system_events_event_type", "system_events", ["event_type"], unique=False)
    op.create_index("ix_system_events_time", "system_events", ["time"], unique=False)

    # ---------------------------
    # channel_cursor / raw_market_events
    # ---------------------------
    op.create_table(
        "channel_cursor",
        sa.Column("channel_id", sa.String(length=64), primary_key=True),
        sa.Column("last_seq", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("last_ingest_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("p99_latency_ms", sa.Integer(), nullable=False, server_default="200"),
        sa.Column(
            "p99_state",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "raw_market_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("api_schema_version", sa.String(length=32), nullable=False),

        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("ths_product", sa.String(length=32), nullable=False),
        sa.Column("ths_function", sa.String(length=128), nullable=False),
        sa.Column("ths_indicator_set", sa.String(length=512), nullable=False),
        sa.Column("ths_params_canonical", sa.String(length=2048), nullable=False),
        sa.Column("ths_errorcode", sa.String(length=64), nullable=False, server_default="0"),
        sa.Column("ths_quota_context", sa.String(length=512), nullable=False, server_default=""),

        sa.Column("source_clock_quality", sa.String(length=32), nullable=False),
        sa.Column("channel_id", sa.String(length=64), nullable=False),
        sa.Column("channel_seq", sa.BigInteger(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),

        sa.Column("data_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingest_ts", sa.DateTime(timezone=True), nullable=False),

        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),

        sa.Column("data_status", sa.String(length=16), nullable=False, server_default="VALID"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_rate", sa.Float(), nullable=False, server_default="1.0"),

        sa.Column("realtime_flag", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("audit_flag", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("research_only", sa.Boolean(), nullable=False, server_default=sa.text("false")),

        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("producer_instance", sa.String(length=64), nullable=False),

        sa.UniqueConstraint("channel_id", "channel_seq", name="uq_event_channel_seq"),
    )
    op.create_index("ix_event_symbol_data_ts", "raw_market_events", ["symbol", "data_ts"], unique=False)
    op.create_index("ix_raw_market_events_data_ts", "raw_market_events", ["data_ts"], unique=False)
    op.create_index("ix_raw_market_events_ingest_ts", "raw_market_events", ["ingest_ts"], unique=False)

    # ---------------------------
    # instrument rules
    # ---------------------------
    op.create_table(
        "instrument_rule_cache",
        sa.Column("symbol", sa.String(length=32), primary_key=True),
        sa.Column("tick_rule_version", sa.String(length=64), nullable=False),
        sa.Column("lot_rule_version", sa.String(length=64), nullable=False),
        sa.Column("tick_size", sa.Float(), nullable=False),
        sa.Column("lot_size", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ---------------------------
    # rule_sets / strategy_contracts
    # ---------------------------
    op.create_table(
        "rule_sets",
        sa.Column("rule_set_version_hash", sa.String(length=64), primary_key=True),
        sa.Column("definition", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "strategy_contracts",
        sa.Column("strategy_contract_hash", sa.String(length=64), primary_key=True),
        sa.Column("definition", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ---------------------------
    # orders / transitions
    # ---------------------------
    op.create_table(
        "orders",
        sa.Column("cid", sa.String(length=64), primary_key=True),
        sa.Column("client_order_id", sa.String(length=80), nullable=False),
        sa.Column("broker_order_id", sa.String(length=80), nullable=True),

        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("order_type", sa.String(length=16), nullable=False),

        sa.Column("limit_price_int64", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("qty_int", sa.BigInteger(), nullable=False),

        sa.Column("tick_rule_version", sa.String(length=64), nullable=False),
        sa.Column("lot_rule_version", sa.String(length=64), nullable=False),
        sa.Column("canonicalization_version", sa.String(length=32), nullable=False),
        sa.Column("metadata_hash", sa.String(length=64), nullable=False),

        sa.Column("state", sa.String(length=20), nullable=False, server_default="CREATED"),
        sa.Column("version_id", sa.Integer(), nullable=False, server_default="1"),

        sa.Column("strategy_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),

        sa.CheckConstraint("qty_int >= 0", name="ck_order_qty_nonneg"),
    )
    op.create_index("ix_orders_symbol", "orders", ["symbol"], unique=False)
    op.create_index("ix_orders_broker_order_id", "orders", ["broker_order_id"], unique=False)
    op.create_index("ix_orders_metadata_hash", "orders", ["metadata_hash"], unique=False)
    op.create_unique_constraint("uq_orders_client_order_id", "orders", ["client_order_id"])

    op.create_table(
        "order_transitions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("cid", sa.String(length=64), sa.ForeignKey("orders.cid", ondelete="CASCADE"), nullable=False),
        sa.Column("transition_id", sa.String(length=64), nullable=False),
        sa.Column("from_state", sa.String(length=20), nullable=False),
        sa.Column("to_state", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("cid", "transition_id", name="uq_order_transition_cid_tid"),
    )
    op.create_index("ix_order_transitions_cid", "order_transitions", ["cid"], unique=False)

    # ---------------------------
    # trade_fills (fills-first immutable target)
    # ---------------------------
    op.create_table(
        "trade_fills",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("broker_fill_id", sa.String(length=128), nullable=False),
        sa.Column("cid", sa.String(length=64), nullable=True),
        sa.Column("broker_order_id", sa.String(length=80), nullable=True),

        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),

        sa.Column("fill_price_int64", sa.BigInteger(), nullable=False),
        sa.Column("fill_qty_int", sa.BigInteger(), nullable=False),
        sa.Column("fill_ts", sa.DateTime(timezone=True), nullable=False),

        sa.Column("fill_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),

        sa.UniqueConstraint("broker_fill_id", name="uq_trade_fills_broker_fill_id"),
        sa.UniqueConstraint("fill_fingerprint", name="uq_trade_fills_fingerprint"),
    )
    op.create_index("ix_trade_fills_symbol", "trade_fills", ["symbol"], unique=False)
    op.create_index("ix_trade_fills_fill_ts", "trade_fills", ["fill_ts"], unique=False)
    op.create_index("ix_trade_fills_cid", "trade_fills", ["cid"], unique=False)
    op.create_index("ix_trade_fills_broker_order_id", "trade_fills", ["broker_order_id"], unique=False)

    # ---------------------------
    # reconcile_snapshots / reconcile_decisions  (必须存在，否则 0003/运行会炸)
    # ---------------------------
    op.create_table(
        "reconcile_snapshots",
        sa.Column("snapshot_id", sa.String(length=64), primary_key=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("anchor_type", sa.String(length=32), nullable=False),
        sa.Column("anchor_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("candidates", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("report_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="OPEN"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reconcile_snapshots_symbol", "reconcile_snapshots", ["symbol"], unique=False)
    op.create_index("ix_reconcile_snapshots_anchor", "reconcile_snapshots", ["anchor_fingerprint"], unique=False)

    op.create_table(
        "reconcile_decisions",
        sa.Column("decision_id", sa.String(length=64), primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.String(length=64),
            sa.ForeignKey("reconcile_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("decided_cid", sa.String(length=64), nullable=False),
        sa.Column("decided_broker_order_id", sa.String(length=80), nullable=True),
        sa.Column("signer_key_id", sa.String(length=64), nullable=False),
        sa.Column("signature", sa.String(length=512), nullable=False),
        sa.Column("prev_decision_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reconcile_decisions_snapshot_id", "reconcile_decisions", ["snapshot_id"], unique=False)

    # ---------------------------
    # decision bundles / signals
    # ---------------------------
    op.create_table(
        "decision_bundles",
        sa.Column("decision_id", sa.String(length=64), primary_key=True),
        sa.Column("cid", sa.String(length=64), nullable=True),
        sa.Column("decision", sa.String(length=8), nullable=False),
        sa.Column("reason_code", sa.String(length=64), nullable=False),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "guard_status",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "data_quality",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("rule_set_version_hash", sa.String(length=64), nullable=False),
        sa.Column("model_snapshot_uuid", sa.String(length=64), nullable=False),
        sa.Column("strategy_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("feature_extractor_version", sa.String(length=64), nullable=False),
        sa.Column("cost_model_version", sa.String(length=64), nullable=False),
        sa.Column("lineage_ref", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_decision_bundles_cid", "decision_bundles", ["cid"], unique=False)

    op.create_table(
        "signals",
        sa.Column("cid", sa.String(length=64), primary_key=True),
        sa.Column("trading_day", sa.String(length=8), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("signal_ts", sa.DateTime(timezone=True), nullable=False),

        sa.Column("nonce", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("intended_qty_or_notional", sa.BigInteger(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.0"),

        sa.Column("rule_set_version_hash", sa.String(length=64), nullable=False),
        sa.Column("strategy_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("model_snapshot_uuid", sa.String(length=64), nullable=False),
        sa.Column("cost_model_version", sa.String(length=64), nullable=False),
        sa.Column("feature_extractor_version", sa.String(length=64), nullable=False),

        sa.Column("lineage_ref", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_signals_symbol_day", "signals", ["symbol", "trading_day"], unique=False)

    # ---------------------------
    # portfolio_positions / trade_log / t1_constraints
    # ---------------------------
    op.create_table(
        "portfolio_positions",
        sa.Column("symbol", sa.String(length=32), primary_key=True),
        sa.Column("current_qty", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("frozen_qty", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("avg_price_int64", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "trade_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("correlation_id", sa.String(length=64), nullable=False),
        sa.Column("cid", sa.String(length=64), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("execution_state", sa.String(length=32), nullable=False),
        sa.Column(
            "feature_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_trade_log_correlation_id", "trade_log", ["correlation_id"], unique=False)
    op.create_index("ix_trade_log_cid", "trade_log", ["cid"], unique=False)
    op.create_index("ix_trade_log_symbol", "trade_log", ["symbol"], unique=False)

    op.create_table(
        "t1_constraints",
        sa.Column("symbol", sa.String(length=32), primary_key=True),
        sa.Column("locked_qty", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ---------------------------
    # training_feature_store / model_snapshots / shadow_runs
    # ---------------------------
    op.create_table(
        "training_feature_store",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),

        sa.Column("data_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingest_ts", sa.DateTime(timezone=True), nullable=False),

        sa.Column("audit_flag", sa.Boolean(), nullable=False),
        sa.Column("realtime_equivalent", sa.Boolean(), nullable=False),

        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("channel_id", sa.String(length=64), nullable=False),
        sa.Column("channel_seq", sa.BigInteger(), nullable=False),
        sa.Column("source_clock_quality", sa.String(length=32), nullable=False),

        sa.Column("feature_extractor_version", sa.String(length=64), nullable=False),
        sa.Column("rule_set_version_hash", sa.String(length=64), nullable=False),
        sa.Column("strategy_contract_hash", sa.String(length=64), nullable=False),

        sa.Column(
            "features",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),

        sa.CheckConstraint("audit_flag = true AND realtime_equivalent = true", name="ck_training_only_realtime_equiv"),
    )
    op.create_index("ix_training_symbol_ts", "training_feature_store", ["symbol", "data_ts"], unique=False)

    op.create_table(
        "model_snapshots",
        sa.Column("model_snapshot_uuid", sa.String(length=64), primary_key=True),
        sa.Column(
            "weights",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "eval_report",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("cost_model_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "shadow_runs",
        sa.Column("run_id", sa.String(length=64), primary_key=True),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("old_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("new_contract_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ---------------------------
    # guardian_keys
    # ---------------------------
    op.create_table(
        "guardian_keys",
        sa.Column("key_id", sa.String(length=64), primary_key=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("public_key_b64", sa.String(length=512), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("guardian_keys")

    op.drop_table("shadow_runs")
    op.drop_table("model_snapshots")

    op.drop_index("ix_training_symbol_ts", table_name="training_feature_store")
    op.drop_table("training_feature_store")

    op.drop_table("t1_constraints")

    op.drop_index("ix_trade_log_symbol", table_name="trade_log")
    op.drop_index("ix_trade_log_cid", table_name="trade_log")
    op.drop_index("ix_trade_log_correlation_id", table_name="trade_log")
    op.drop_table("trade_log")

    op.drop_table("portfolio_positions")

    op.drop_index("ix_signals_symbol_day", table_name="signals")
    op.drop_table("signals")

    op.drop_index("ix_decision_bundles_cid", table_name="decision_bundles")
    op.drop_table("decision_bundles")

    op.drop_index("ix_reconcile_decisions_snapshot_id", table_name="reconcile_decisions")
    op.drop_table("reconcile_decisions")

    op.drop_index("ix_reconcile_snapshots_anchor", table_name="reconcile_snapshots")
    op.drop_index("ix_reconcile_snapshots_symbol", table_name="reconcile_snapshots")
    op.drop_table("reconcile_snapshots")

    op.drop_index("ix_trade_fills_broker_order_id", table_name="trade_fills")
    op.drop_index("ix_trade_fills_cid", table_name="trade_fills")
    op.drop_index("ix_trade_fills_fill_ts", table_name="trade_fills")
    op.drop_index("ix_trade_fills_symbol", table_name="trade_fills")
    op.drop_table("trade_fills")

    op.drop_index("ix_order_transitions_cid", table_name="order_transitions")
    op.drop_table("order_transitions")

    op.drop_constraint("uq_orders_client_order_id", "orders", type_="unique")
    op.drop_index("ix_orders_metadata_hash", table_name="orders")
    op.drop_index("ix_orders_broker_order_id", table_name="orders")
    op.drop_index("ix_orders_symbol", table_name="orders")
    op.drop_table("orders")

    op.drop_table("strategy_contracts")
    op.drop_table("rule_sets")

    op.drop_table("instrument_rule_cache")

    op.drop_index("ix_raw_market_events_ingest_ts", table_name="raw_market_events")
    op.drop_index("ix_raw_market_events_data_ts", table_name="raw_market_events")
    op.drop_index("ix_event_symbol_data_ts", table_name="raw_market_events")
    op.drop_table("raw_market_events")

    op.drop_table("channel_cursor")

    op.drop_index("ix_system_events_time", table_name="system_events")
    op.drop_index("ix_system_events_event_type", table_name="system_events")
    op.drop_index("ix_system_events_type_time", table_name="system_events")
    op.drop_table("system_events")

    op.drop_table("system_status")
