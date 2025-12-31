"""anchors + frozen versions + self check gate fields

Revision ID: 0004_anchors_frozen_versions_selfcheck
Revises: 0003_immutability_nonce_outbox_ambiguous
Create Date: 2025-12-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_anchors_frozen_versions_selfcheck"
down_revision = "0003_immutability_nonce_outbox_ambiguous"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "daily_frozen_versions",
        sa.Column("trading_day", sa.String(length=8), primary_key=True),
        sa.Column("rule_set_version_hash", sa.String(length=64), nullable=False),
        sa.Column("strategy_contract_hash", sa.String(length=64), nullable=False),
        sa.Column("model_snapshot_uuid", sa.String(length=64), nullable=False),
        sa.Column("cost_model_version", sa.String(length=64), nullable=False),
        sa.Column("canonicalization_version", sa.String(length=32), nullable=False),
        sa.Column("feature_extractor_version", sa.String(length=64), nullable=False),
        sa.Column("report_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "order_anchors",
        sa.Column("cid", sa.String(length=64), primary_key=True),
        sa.Column("client_order_id", sa.String(length=80), nullable=False),
        sa.Column("broker_order_id", sa.String(length=80), nullable=True),
        sa.Column("request_uuid", sa.String(length=64), nullable=False),
        sa.Column("ack_hash", sa.String(length=64), nullable=False),
        sa.Column("raw_request_hash", sa.String(length=64), nullable=False),
        sa.Column("raw_response_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_unique_constraint("uq_order_anchors_request_uuid", "order_anchors", ["request_uuid"])
    op.create_index("ix_order_anchor_broker", "order_anchors", ["broker_order_id"], unique=False)
    op.create_index("ix_order_anchor_client", "order_anchors", ["client_order_id"], unique=False)

    op.add_column("system_status", sa.Column("last_self_check_report_hash", sa.String(length=64), nullable=True))
    op.add_column("system_status", sa.Column("last_self_check_time", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("system_status", "last_self_check_time")
    op.drop_column("system_status", "last_self_check_report_hash")

    op.drop_index("ix_order_anchor_client", table_name="order_anchors")
    op.drop_index("ix_order_anchor_broker", table_name="order_anchors")
    op.drop_constraint("uq_order_anchors_request_uuid", "order_anchors", type_="unique")
    op.drop_table("order_anchors")

    op.drop_table("daily_frozen_versions")
