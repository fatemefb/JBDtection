"""Add user management tables (users, user_sessions, audit_logs, io_assignment_presets).

Revision ID: 20260722_0002
Revises: 20260209_0001
Create Date: 2026-07-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260722_0002"
down_revision = "20260209_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Create new ENUM types idempotently
    op.execute(
        """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('admin','engineer','viewer');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'session_status') THEN
        CREATE TYPE session_status AS ENUM ('active','expired','logged_out','revoked');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'audit_action') THEN
        CREATE TYPE audit_action AS ENUM (
            'login','login_failed','logout',
            'user_create','user_update','user_delete',
            'user_activate','user_deactivate',
            'preset_save','preset_load','preset_delete',
            'run_start','run_finalize','permission_denied'
        );
    END IF;
END $$;
"""
    )

    # 2) Reference ENUM types without recreating them
    user_role = postgresql.ENUM(
        "admin", "engineer", "viewer",
        name="user_role",
        create_type=False,
    )
    session_status = postgresql.ENUM(
        "active", "expired", "logged_out", "revoked",
        name="session_status",
        create_type=False,
    )
    audit_action = postgresql.ENUM(
        "login", "login_failed", "logout",
        "user_create", "user_update", "user_delete",
        "user_activate", "user_deactivate",
        "preset_save", "preset_load", "preset_delete",
        "run_start", "run_finalize", "permission_denied",
        name="audit_action",
        create_type=False,
    )

    # 3) Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("username", sa.String(128), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(60), nullable=False),
        sa.Column("role", user_role, nullable=False, server_default="engineer"),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("failed_login_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_ip", sa.String(45), nullable=True),
        sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_by", sa.String(128), nullable=True),
    )
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_role", "users", ["role"])
    op.create_index("ix_users_active", "users", ["is_active"])

    # 4) Create user_sessions table
    op.create_table(
        "user_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_token", sa.String(128), nullable=False, unique=True),
        sa.Column("status", session_status, nullable=False, server_default="active"),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("logged_out_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by", sa.String(128), nullable=True),
        sa.Column("meta", postgresql.JSONB, nullable=True),
    )
    op.create_index("ix_user_sessions_session_token", "user_sessions", ["session_token"])
    op.create_index("ix_user_sessions_user_status", "user_sessions", ["user_id", "status"])
    op.create_index("ix_user_sessions_expires", "user_sessions", ["expires_at"])

    # 5) Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("username", sa.String(128), nullable=True),
        sa.Column("action", audit_action, nullable=False),
        sa.Column("target_username", sa.String(128), nullable=True),
        sa.Column("target_resource", sa.String(255), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("details", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_audit_logs_username", "audit_logs", ["username"])
    op.create_index("ix_audit_logs_user_action", "audit_logs", ["user_id", "action"])
    op.create_index("ix_audit_logs_action_created", "audit_logs", ["action", "created_at"])
    op.create_index("ix_audit_logs_created", "audit_logs", ["created_at"])

    # 6) Create io_assignment_presets table
    op.create_table(
        "io_assignment_presets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("project_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("type_count", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("has_directions", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("cabinet_plan", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("cabinet_dimensions", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("card_catalog", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "project_name", name="uq_preset_user_project"),
    )
    op.create_index("ix_presets_user", "io_assignment_presets", ["user_id"])
    op.create_index("ix_presets_project", "io_assignment_presets", ["project_name"])

    # 7) Migrate Run.initiated_by references to be consistent with users table.
    # The initiated_by column stores usernames as strings — we DON'T migrate
    # these to FK because runs may have been created by users that no longer
    # exist. The username string is enough for filtering.
    # No data migration needed.

    # 8) Seed default admin user (password: admin123 — must be changed on first login)
    # bcrypt hash for "admin123" with rounds=12
    op.execute(
        """
        INSERT INTO users (id, username, password_hash, role, display_name, is_active, created_by)
        VALUES (
            gen_random_uuid(),
            'admin',
            '$2b$12$TiW7AAo5rRNRwB6EYBe1ce6chZhEq61ZjMDBQEUE1MLobrk8bcT7C',
            'admin',
            'Administrator',
            true,
            'system'
        )
        ON CONFLICT (username) DO NOTHING;
        """
    )


def downgrade() -> None:
    op.drop_table("io_assignment_presets")
    op.drop_table("audit_logs")
    op.drop_table("user_sessions")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS audit_action")
    op.execute("DROP TYPE IF EXISTS session_status")
    op.execute("DROP TYPE IF EXISTS user_role")
