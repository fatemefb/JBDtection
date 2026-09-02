"""Initial DB schema for DB-centric pipeline."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260209_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Needed for gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # 1) Create ENUM types idempotently (prevents DuplicateObject)
    op.execute(
        """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'run_status') THEN
        CREATE TYPE run_status AS ENUM ('pending','processing','review','finalized','failed','archived');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'issue_severity') THEN
        CREATE TYPE issue_severity AS ENUM ('error','warning','info');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'issue_status') THEN
        CREATE TYPE issue_status AS ENUM ('open','resolved','muted');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'artifact_type') THEN
        CREATE TYPE artifact_type AS ENUM ('final_excel','annotated_pdf','zip_bundle','unmatched_excel','report_json','raw_pdfs');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'uploaded_file_type') THEN
        CREATE TYPE uploaded_file_type AS ENUM ('pdf','excel','other');
    END IF;
END $$;
"""
    )

    # 2) Reference existing ENUM types without letting SQLAlchemy create them
    run_status = postgresql.ENUM(
        "pending",
        "processing",
        "review",
        "finalized",
        "failed",
        "archived",
        name="run_status",
        create_type=False,
    )

    issue_severity = postgresql.ENUM(
        "error",
        "warning",
        "info",
        name="issue_severity",
        create_type=False,
    )

    issue_status = postgresql.ENUM(
        "open",
        "resolved",
        "muted",
        name="issue_status",
        create_type=False,
    )

    artifact_type = postgresql.ENUM(
        "final_excel",
        "annotated_pdf",
        "zip_bundle",
        "unmatched_excel",
        "report_json",
        "raw_pdfs",
        name="artifact_type",
        create_type=False,
    )

    uploaded_file_type = postgresql.ENUM(
        "pdf",
        "excel",
        "other",
        name="uploaded_file_type",
        create_type=False,
    )

    op.create_table(
        "projects",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("project_name", sa.String(length=255), nullable=False),
        sa.Column("project_hash", sa.String(length=128), nullable=True),
        sa.Column("encoded_name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_finalized_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("project_name", name="uq_project_name"),
        sa.UniqueConstraint("project_hash", name="uq_project_hash"),
    )
    op.create_index("ix_projects_project_name", "projects", ["project_name"], unique=True)

    op.create_table(
        "runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", run_status, nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(length=50), nullable=False, server_default="import"),
        sa.Column("input_hash", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("initiated_by", sa.String(length=128), nullable=True),
        sa.Column("reuse_of_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("runs.id"), nullable=True),
        sa.Column("keep_latest_only", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_runs_project_status", "runs", ["project_id", "status"], unique=False)
    op.create_index("ix_runs_input_hash", "runs", ["input_hash"], unique=False)

    op.create_table(
        "uploaded_files",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("file_type", uploaded_file_type, nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("stored_name", sa.String(length=255), nullable=False),
        sa.Column("file_hash", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("file_hash", "file_type", "project_id", name="uq_uploaded_file_hash_project_type"),
    )
    op.create_index("ix_uploaded_files_project", "uploaded_files", ["project_id"], unique=False)
    op.create_index("ix_uploaded_files_run", "uploaded_files", ["run_id"], unique=False)

    op.create_table(
        "pdf_pages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "pdf_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("uploaded_files.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_index", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("text_content", sa.Text(), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("pdf_file_id", "page_index", name="uq_pdf_page"),
    )
    op.create_index("ix_pdf_pages_run", "pdf_pages", ["run_id"], unique=False)

    op.create_table(
        "io_list_rows",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("jb", sa.String(length=128), nullable=True),
        sa.Column("io_type", sa.String(length=64), nullable=True),
        sa.Column("safety", sa.String(length=64), nullable=True),
        sa.Column("location", sa.String(length=128), nullable=True),
        sa.Column("terminal1", sa.String(length=128), nullable=True),
        sa.Column("terminal2", sa.String(length=128), nullable=True),
        sa.Column("src", sa.String(length=256), nullable=True),
        sa.Column("normalized_tag", sa.String(length=256), nullable=True),
        sa.Column("match_status", sa.String(length=32), nullable=True),
        sa.Column("matched_tag_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("raw_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("issue_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("run_id", "row_index", name="uq_io_row_idx"),
    )
    op.create_index("ix_io_rows_project", "io_list_rows", ["project_id"], unique=False)

    op.create_table(
        "tag_occurrences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "pdf_page_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pdf_pages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag_text", sa.String(length=256), nullable=False),
        sa.Column("normalized_tag", sa.String(length=256), nullable=True),
        sa.Column("bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("match_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("matched_io_row_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_tag_occurrences_normtag", "tag_occurrences", ["normalized_tag"], unique=False)

    op.create_table(
        "issues",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "io_list_row_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("io_list_rows.id"),
            nullable=True,
        ),
        sa.Column(
            "tag_occurrence_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tag_occurrences.id"),
            nullable=True,
        ),
        sa.Column("pdf_page_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("pdf_pages.id"), nullable=True),
        sa.Column("severity", issue_severity, nullable=False, server_default="warning"),
        sa.Column("status", issue_status, nullable=False, server_default="open"),
        sa.Column("code", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_issues_run", "issues", ["run_id"], unique=False)
    op.create_index("ix_issues_project", "issues", ["project_id"], unique=False)

    op.create_table(
        "run_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("level", sa.String(length=16), nullable=False, server_default="info"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_run_logs_run_created", "run_logs", ["run_id", "created_at"], unique=False)

    op.create_table(
        "export_artifacts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("artifact_type", artifact_type, nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("file_hash", sa.String(length=128), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.UniqueConstraint("run_id", "artifact_type", name="uq_artifact_per_run_type"),
    )

    # add FK from projects to runs after runs table exists
    op.create_foreign_key(
        "projects_last_finalized_run_fk",
        source_table="projects",
        referent_table="runs",
        local_cols=["last_finalized_run_id"],
        remote_cols=["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("projects_last_finalized_run_fk", "projects", type_="foreignkey")
    op.drop_table("export_artifacts")
    op.drop_table("run_logs")
    op.drop_table("issues")
    op.drop_table("tag_occurrences")
    op.drop_table("io_list_rows")
    op.drop_table("pdf_pages")
    op.drop_table("uploaded_files")
    op.drop_table("runs")
    op.drop_table("projects")

    op.execute("DROP TYPE IF EXISTS artifact_type")
    op.execute("DROP TYPE IF EXISTS issue_status")
    op.execute("DROP TYPE IF EXISTS issue_severity")
    op.execute("DROP TYPE IF EXISTS run_status")
    op.execute("DROP TYPE IF EXISTS uploaded_file_type")
