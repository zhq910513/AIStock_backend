"""channel cursor fidelity + source_fidelity_daily

Revision ID: 0002_channel_fidelity_and_daily
Revises: 0001_init_qee_s3
Create Date: 2025-12-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_channel_fidelity_and_daily"
down_revision = "0001_init_qee_s3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("channel_cursor", sa.Column("fidelity_score", sa.Float(), nullable=False, server_default="1.0"))
    op.add_column("channel_cursor", sa.Column("fidelity_low_streak", sa.Integer(), nullable=False, server_default="0"))

    op.create_table(
        "source_fidelity_daily",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("data_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("channel_id_a", sa.String(length=64), nullable=False),
        sa.Column("channel_id_b", sa.String(length=64), nullable=False),
        sa.Column("close_a", sa.BigInteger(), nullable=False),
        sa.Column("close_b", sa.BigInteger(), nullable=False),
        sa.Column("abs_diff", sa.BigInteger(), nullable=False),
        sa.Column("threshold", sa.BigInteger(), nullable=False),
        sa.Column("fidelity_score_before", sa.Float(), nullable=False),
        sa.Column("fidelity_score_after", sa.Float(), nullable=False),
        sa.Column("action_taken", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_fidelity_symbol_ts", "source_fidelity_daily", ["symbol", "data_ts"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_fidelity_symbol_ts", table_name="source_fidelity_daily")
    op.drop_table("source_fidelity_daily")
    op.drop_column("channel_cursor", "fidelity_low_streak")
    op.drop_column("channel_cursor", "fidelity_score")
