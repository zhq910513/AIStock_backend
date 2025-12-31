"""immutability + nonce/outbox/symbol_lock (fixed)

Revision ID: 0003_immutability_nonce_outbox_ambiguous
Revises: 0002_channel_fidelity_and_daily
Create Date: 2025-12-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_immutability_nonce_outbox_ambiguous"
down_revision = "0002_channel_fidelity_and_daily"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # nonce_cursor
    op.create_table(
        "nonce_cursor",
        sa.Column("trading_day", sa.String(length=8), primary_key=True),
        sa.Column("symbol", sa.String(length=32), primary_key=True),
        sa.Column("strategy_id", sa.String(length=64), primary_key=True),
        sa.Column("last_nonce", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # symbol_locks
    op.create_table(
        "symbol_locks",
        sa.Column("symbol", sa.String(length=32), primary_key=True),
        sa.Column("locked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("lock_reason", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("lock_ref", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # outbox_events
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("dedupe_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="PENDING"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("last_error", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint("uq_outbox_dedupe_key", "outbox_events", ["dedupe_key"])
    op.create_index("ix_outbox_event_type", "outbox_events", ["event_type"], unique=False)
    op.create_index("ix_outbox_created_at", "outbox_events", ["created_at"], unique=False)

    # --- immutability triggers (Postgres) ---
    op.execute(
        """
        CREATE OR REPLACE FUNCTION qee_immutable_guard()
        RETURNS trigger AS $$
        BEGIN
          RAISE EXCEPTION 'immutable_table_violation: %', TG_TABLE_NAME;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    for tbl in ("system_events", "raw_market_events", "trade_fills"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{tbl}_no_update ON {tbl};")
        op.execute(f"DROP TRIGGER IF EXISTS trg_{tbl}_no_delete ON {tbl};")
        op.execute(
            f"""
            CREATE TRIGGER trg_{tbl}_no_update
            BEFORE UPDATE ON {tbl}
            FOR EACH ROW EXECUTE FUNCTION qee_immutable_guard();
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER trg_{tbl}_no_delete
            BEFORE DELETE ON {tbl}
            FOR EACH ROW EXECUTE FUNCTION qee_immutable_guard();
            """
        )


def downgrade() -> None:
    for tbl in ("system_events", "raw_market_events", "trade_fills"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{tbl}_no_update ON {tbl};")
        op.execute(f"DROP TRIGGER IF EXISTS trg_{tbl}_no_delete ON {tbl};")
    op.execute("DROP FUNCTION IF EXISTS qee_immutable_guard;")

    op.drop_index("ix_outbox_created_at", table_name="outbox_events")
    op.drop_index("ix_outbox_event_type", table_name="outbox_events")
    op.drop_constraint("uq_outbox_dedupe_key", "outbox_events", type_="unique")
    op.drop_table("outbox_events")

    op.drop_table("symbol_locks")
    op.drop_table("nonce_cursor")
