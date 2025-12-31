"""orders add last_transition_id for idempotent state machine

Revision ID: 0006_order_last_transition_id
Revises: 0005_schema_completion_core_tables
Create Date: 2025-12-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0006_order_last_transition_id"
down_revision = "0005_schema_completion_core_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orders", sa.Column("last_transition_id", sa.String(length=64), nullable=True))
    op.create_index("ix_orders_last_transition_id", "orders", ["last_transition_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_orders_last_transition_id", table_name="orders")
    op.drop_column("orders", "last_transition_id")
