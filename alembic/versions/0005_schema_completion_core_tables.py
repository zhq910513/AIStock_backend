"""outbox retry window available_at

Revision ID: 0005_outbox_retry_available_at
Revises: 0004_anchors_frozen_versions_selfcheck
Create Date: 2025-12-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_outbox_retry_available_at"
down_revision = "0004_anchors_frozen_versions_selfcheck"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("outbox_events", sa.Column("available_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE outbox_events SET available_at = created_at WHERE available_at IS NULL;")
    op.alter_column("outbox_events", "available_at", nullable=False)
    op.create_index("ix_outbox_available_at", "outbox_events", ["available_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_outbox_available_at", table_name="outbox_events")
    op.drop_column("outbox_events", "available_at")
