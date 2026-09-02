from __future__ import annotations

import enum
from datetime import datetime
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, BYTEA
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow():
    return datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS — User Management (NEW)
# ═══════════════════════════════════════════════════════════════════════════
class UserRole(str, enum.Enum):
    """User role hierarchy: admin > engineer > viewer."""
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"


class SessionStatus(str, enum.Enum):
    """Status of a user session."""
    ACTIVE = "active"
    EXPIRED = "expired"
    LOGGED_OUT = "logged_out"
    REVOKED = "revoked"


class AuditAction(str, enum.Enum):
    """Action types recorded in the audit log."""
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_ACTIVATE = "user_activate"
    USER_DEACTIVATE = "user_deactivate"
    PRESET_SAVE = "preset_save"
    PRESET_LOAD = "preset_load"
    PRESET_DELETE = "preset_delete"
    RUN_START = "run_start"
    RUN_FINALIZE = "run_finalize"
    PERMISSION_DENIED = "permission_denied"


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS — Existing (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
class RunStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    REVIEW = "review"
    FINALIZED = "finalized"
    FAILED = "failed"
    ARCHIVED = "archived"


class IssueSeverity(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueStatus(str, enum.Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    MUTED = "muted"


class ArtifactType(str, enum.Enum):
    FINAL_EXCEL = "final_excel"
    ANNOTATED_PDF = "annotated_pdf"
    ZIP_BUNDLE = "zip_bundle"
    UNMATCHED_EXCEL = "unmatched_excel"
    REPORT_JSON = "report_json"
    RAW_PDFS = "raw_pdfs"


class UploadedFileType(str, enum.Enum):
    PDF = "pdf"
    EXCEL = "excel"
    OTHER = "other"


# ═══════════════════════════════════════════════════════════════════════════
# EXISTING TABLES (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
class Project(Base):
    __tablename__ = "projects"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_name = sa.Column(sa.String(255), nullable=False, unique=True, index=True)
    project_hash = sa.Column(sa.String(128), nullable=True, unique=True)
    encoded_name = sa.Column(sa.String(255), nullable=True)
    is_active = sa.Column(sa.Boolean, nullable=False, default=True)
    last_finalized_run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id"), nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    updated_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), onupdate=sa.func.now(), nullable=False)

    runs = relationship(
        "Run",
        back_populates="project",
        cascade="all, delete-orphan",
        foreign_keys="Run.project_id",
    )
    last_finalized_run = relationship("Run", foreign_keys=[last_finalized_run_id], post_update=True)


class Run(Base):
    __tablename__ = "runs"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    status = sa.Column(
        sa.Enum(
            RunStatus,
            name="run_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
        default=RunStatus.PENDING,
    )
    stage = sa.Column(sa.String(50), nullable=False, default="import")
    input_hash = sa.Column(sa.String(128), nullable=True, index=True)
    notes = sa.Column(sa.Text, nullable=True)
    initiated_by = sa.Column(sa.String(128), nullable=True)
    reuse_of_run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id"), nullable=True)
    keep_latest_only = sa.Column(sa.Boolean, default=True, nullable=False)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    started_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    finished_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    updated_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), onupdate=sa.func.now(), nullable=False)

    project = relationship("Project", back_populates="runs", foreign_keys=[project_id])
    reused_from = relationship("Run", remote_side=[id])
    files = relationship("UploadedFile", back_populates="run", cascade="all, delete-orphan")
    io_rows = relationship("IOListRow", back_populates="run", cascade="all, delete-orphan")
    tag_occurrences = relationship("TagOccurrence", back_populates="run", cascade="all, delete-orphan")
    issues = relationship("Issue", back_populates="run", cascade="all, delete-orphan")
    logs = relationship("RunLog", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("ExportArtifact", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        sa.Index("ix_runs_project_status", "project_id", "status"),
    )


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=True)
    file_type = sa.Column(
        sa.Enum(
            UploadedFileType,
            name="uploaded_file_type",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
    )
    original_name = sa.Column(sa.String(255), nullable=False)
    stored_name = sa.Column(sa.String(255), nullable=False)
    file_hash = sa.Column(sa.String(128), nullable=False)
    size_bytes = sa.Column(sa.BigInteger, nullable=True)
    mime_type = sa.Column(sa.String(128), nullable=True)
    storage_path = sa.Column(sa.Text, nullable=True)
    content = sa.Column(BYTEA, nullable=True)
    page_count = sa.Column(sa.Integer, nullable=True)
    meta = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)

    run = relationship("Run", back_populates="files")
    project = relationship("Project")
    pages = relationship("PDFPage", back_populates="pdf_file", cascade="all, delete-orphan")

    __table_args__ = (
        sa.UniqueConstraint("file_hash", "file_type", "project_id", name="uq_uploaded_file_hash_project_type"),
        sa.Index("ix_uploaded_files_project", "project_id"),
        sa.Index("ix_uploaded_files_run", "run_id"),
    )


class PDFPage(Base):
    __tablename__ = "pdf_pages"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    pdf_file_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("uploaded_files.id", ondelete="CASCADE"), nullable=False)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    page_index = sa.Column(sa.Integer, nullable=False)
    width = sa.Column(sa.Integer, nullable=True)
    height = sa.Column(sa.Integer, nullable=True)
    text_content = sa.Column(sa.Text, nullable=True)
    image_path = sa.Column(sa.Text, nullable=True)
    image_bytes = sa.Column(BYTEA, nullable=True)
    meta = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)

    pdf_file = relationship("UploadedFile", back_populates="pages")
    run = relationship("Run")
    tag_occurrences = relationship("TagOccurrence", back_populates="pdf_page", cascade="all, delete-orphan")

    __table_args__ = (
        sa.UniqueConstraint("pdf_file_id", "page_index", name="uq_pdf_page"),
        sa.Index("ix_pdf_pages_run", "run_id"),
    )


class IOListRow(Base):
    __tablename__ = "io_list_rows"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    row_index = sa.Column(sa.Integer, nullable=False)
    jb = sa.Column(sa.String(128), nullable=True)
    io_type = sa.Column(sa.String(64), nullable=True)
    safety = sa.Column(sa.String(64), nullable=True)
    location = sa.Column(sa.String(128), nullable=True)
    terminal1 = sa.Column(sa.String(128), nullable=True)
    terminal2 = sa.Column(sa.String(128), nullable=True)
    src = sa.Column(sa.String(256), nullable=True)
    normalized_tag = sa.Column(sa.String(256), nullable=True)
    match_status = sa.Column(sa.String(32), nullable=True)
    matched_tag_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("tag_occurrences.id"), nullable=True)
    raw_json = sa.Column(JSONB, nullable=True)
    issue_count = sa.Column(sa.Integer, nullable=False, default=0)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    updated_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), onupdate=sa.func.now(), nullable=False)

    run = relationship("Run", back_populates="io_rows")
    project = relationship("Project")
    matched_tag = relationship("TagOccurrence", foreign_keys=[matched_tag_id])
    issues = relationship("Issue", back_populates="io_row")

    __table_args__ = (
        sa.UniqueConstraint("run_id", "row_index", name="uq_io_row_idx"),
        sa.Index("ix_io_rows_project", "project_id"),
    )


class TagOccurrence(Base):
    __tablename__ = "tag_occurrences"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    pdf_page_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("pdf_pages.id", ondelete="CASCADE"), nullable=False)
    tag_text = sa.Column(sa.String(256), nullable=False)
    normalized_tag = sa.Column(sa.String(256), nullable=True, index=True)
    bbox = sa.Column(JSONB, nullable=True)
    confidence = sa.Column(sa.Numeric(5, 4), nullable=True)
    match_score = sa.Column(sa.Numeric(5, 4), nullable=True)
    matched_io_row_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("io_list_rows.id"), nullable=True)
    attributes = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)

    run = relationship("Run", back_populates="tag_occurrences")
    project = relationship("Project")
    pdf_page = relationship("PDFPage", back_populates="tag_occurrences")
    matched_io_row = relationship("IOListRow", foreign_keys=[matched_io_row_id])
    issues = relationship("Issue", back_populates="tag_occurrence")

    __table_args__ = (
        sa.Index("ix_tag_occurrences_normtag", "normalized_tag"),
    )


class Issue(Base):
    __tablename__ = "issues"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    io_list_row_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("io_list_rows.id"), nullable=True)
    tag_occurrence_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("tag_occurrences.id"), nullable=True)
    pdf_page_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("pdf_pages.id"), nullable=True)
    severity = sa.Column(
        sa.Enum(
            IssueSeverity,
            name="issue_severity",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
        default=IssueSeverity.WARNING,
    )
    status = sa.Column(
        sa.Enum(
            IssueStatus,
            name="issue_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
        default=IssueStatus.OPEN,
    )
    code = sa.Column(sa.String(64), nullable=True)
    message = sa.Column(sa.Text, nullable=False)
    details = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    resolved_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    resolved_by = sa.Column(sa.String(128), nullable=True)

    run = relationship("Run", back_populates="issues")
    project = relationship("Project")
    io_row = relationship("IOListRow", back_populates="issues")
    tag_occurrence = relationship("TagOccurrence", back_populates="issues")
    pdf_page = relationship("PDFPage")

    __table_args__ = (
        sa.Index("ix_issues_run", "run_id"),
        sa.Index("ix_issues_project", "project_id"),
    )


class RunLog(Base):
    __tablename__ = "run_logs"

    id = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    level = sa.Column(sa.String(16), nullable=False, default="info")
    message = sa.Column(sa.Text, nullable=False)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)

    run = relationship("Run", back_populates="logs")
    project = relationship("Project")

    __table_args__ = (
        sa.Index("ix_run_logs_run_created", "run_id", "created_at"),
    )


class ExportArtifact(Base):
    __tablename__ = "export_artifacts"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    project_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    artifact_type = sa.Column(
        sa.Enum(
            ArtifactType,
            name="artifact_type",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
    )
    storage_path = sa.Column(sa.Text, nullable=True)
    file_hash = sa.Column(sa.String(128), nullable=True)
    size_bytes = sa.Column(sa.BigInteger, nullable=True)
    mime_type = sa.Column(sa.String(128), nullable=True)
    ready_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    meta = sa.Column(JSONB, nullable=True)

    run = relationship("Run", back_populates="artifacts")
    project = relationship("Project")

    __table_args__ = (
        sa.UniqueConstraint("run_id", "artifact_type", name="uq_artifact_per_run_type"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT TABLES (NEW — added 2026-07-22)
# ═══════════════════════════════════════════════════════════════════════════
# Four tables added for multi-user support:
#   1. users                  — user accounts (bcrypt password hashes)
#   2. user_sessions          — server-side session tracking (force-logout capable)
#   3. audit_logs             — append-only audit trail (compliance + debugging)
#   4. io_assignment_presets  — per-user cabinet configuration presets
# ═══════════════════════════════════════════════════════════════════════════


class User(Base):
    """
    User account. Replaces the hardcoded VALID_USERS dict.

    Passwords are stored as bcrypt hashes (60 chars). The password_hash
    column is set via the helper method `set_password()` — never write
    plaintext to this column.

    The `is_active` flag allows deactivating a user without deleting them
    (preserves audit trail integrity). Inactive users cannot log in.
    """
    __tablename__ = "users"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = sa.Column(sa.String(128), nullable=False, unique=True, index=True)
    password_hash = sa.Column(sa.String(60), nullable=False)  # bcrypt hash
    role = sa.Column(
        sa.Enum(
            UserRole,
            name="user_role",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
        default=UserRole.ENGINEER,
    )
    display_name = sa.Column(sa.String(255), nullable=True)
    email = sa.Column(sa.String(255), nullable=True, index=True)
    is_active = sa.Column(sa.Boolean, nullable=False, default=True)
    failed_login_count = sa.Column(sa.Integer, nullable=False, default=0)
    locked_until = sa.Column(sa.DateTime(timezone=True), nullable=True)
    last_login_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    last_login_ip = sa.Column(sa.String(45), nullable=True)  # IPv6 max length
    password_changed_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    updated_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    created_by = sa.Column(sa.String(128), nullable=True)  # username of admin who created this user

    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_entries = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    presets = relationship("IOAssignmentPreset", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        sa.Index("ix_users_role", "role"),
        sa.Index("ix_users_active", "is_active"),
    )

    def set_password(self, plaintext: str) -> None:
        """Hash a plaintext password and store the hash. Requires bcrypt."""
        import bcrypt
        if not plaintext:
            raise ValueError("Password cannot be empty")
        if len(plaintext) < 4:
            raise ValueError("Password must be at least 4 characters")
        self.password_hash = bcrypt.hashpw(
            plaintext.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        ).decode("utf-8")
        self.password_changed_at = datetime.utcnow()

    def check_password(self, plaintext: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        if not self.password_hash or not plaintext:
            return False
        try:
            import bcrypt
            return bcrypt.checkpw(
                plaintext.encode("utf-8"),
                self.password_hash.encode("utf-8"),
            )
        except Exception:
            return False

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Serialize for API responses. Excludes password_hash by default."""
        return {
            "id": str(self.id) if self.id else None,
            "username": self.username,
            "role": self.role.value if self.role else None,
            "display_name": self.display_name,
            "email": self.email,
            "is_active": self.is_active,
            "failed_login_count": self.failed_login_count,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "last_login_ip": self.last_login_ip,
            "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            **({"password_hash": self.password_hash} if include_sensitive else {}),
        }


class UserSession(Base):
    """
    Server-side session tracking. Each login creates a session row; logout
    marks it as logged_out. Allows the admin to see who's currently online
    and to revoke active sessions (force-logout a user).
    """
    __tablename__ = "user_sessions"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token = sa.Column(sa.String(128), nullable=False, unique=True, index=True)
    status = sa.Column(
        sa.Enum(
            SessionStatus,
            name="session_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
        default=SessionStatus.ACTIVE,
    )
    ip_address = sa.Column(sa.String(45), nullable=True)
    user_agent = sa.Column(sa.Text, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    last_activity_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    expires_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    logged_out_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    revoked_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    revoked_by = sa.Column(sa.String(128), nullable=True)  # username of admin who revoked
    meta = sa.Column(JSONB, nullable=True)

    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        sa.Index("ix_user_sessions_user_status", "user_id", "status"),
        sa.Index("ix_user_sessions_expires", "expires_at"),
    )


class AuditLog(Base):
    """
    Append-only audit log. Records every security-relevant action:
    logins, user management, preset save/load/delete, permission denials.
    Used for compliance and for debugging "who did what" questions.
    """
    __tablename__ = "audit_logs"

    id = sa.Column(sa.BigInteger, primary_key=True, autoincrement=True)
    user_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    username = sa.Column(sa.String(128), nullable=True, index=True)  # denormalized for fast queries even after user deletion
    action = sa.Column(
        sa.Enum(
            AuditAction,
            name="audit_action",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
        nullable=False,
    )
    target_username = sa.Column(sa.String(128), nullable=True)  # for user-management actions
    target_resource = sa.Column(sa.String(255), nullable=True)  # e.g. preset name, run ID
    ip_address = sa.Column(sa.String(45), nullable=True)
    user_agent = sa.Column(sa.Text, nullable=True)
    success = sa.Column(sa.Boolean, nullable=False, default=True)
    message = sa.Column(sa.Text, nullable=True)
    details = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)

    user = relationship("User", back_populates="audit_entries")

    __table_args__ = (
        sa.Index("ix_audit_logs_user_action", "user_id", "action"),
        sa.Index("ix_audit_logs_action_created", "action", "created_at"),
        sa.Index("ix_audit_logs_created", "created_at"),
    )


class IOAssignmentPreset(Base):
    """
    Per-user cabinet configuration preset. Stores the full DimensionMode
    state (cabinet_dimensions, cabinet_plan, card_catalog) so users can save
    a project's settings and reload them later.

    Owned by a user (user_id). Admin users can see all presets; non-admin
    users see only their own presets. Project name is unique per user.
    """
    __tablename__ = "io_assignment_presets"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    project_name = sa.Column(sa.String(255), nullable=False)
    description = sa.Column(sa.Text, nullable=True)
    type_count = sa.Column(sa.Integer, nullable=False, default=1)
    has_directions = sa.Column(sa.Boolean, nullable=False, default=True)
    cabinet_plan = sa.Column(JSONB, nullable=False, default="[]")
    cabinet_dimensions = sa.Column(JSONB, nullable=False, default="{}")
    card_catalog = sa.Column(JSONB, nullable=False, default="{}")
    created_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), nullable=False)
    updated_at = sa.Column(sa.DateTime(timezone=True), default=sa.func.now(), onupdate=sa.func.now(), nullable=False)

    user = relationship("User", back_populates="presets")

    __table_args__ = (
        sa.UniqueConstraint("user_id", "project_name", name="uq_preset_user_project"),
        sa.Index("ix_presets_user", "user_id"),
        sa.Index("ix_presets_project", "project_name"),
    )

    def to_dict(self, include_config: bool = True) -> dict:
        """Serialize for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "project_name": self.project_name,
            "description": self.description,
            "type_count": self.type_count,
            "has_directions": self.has_directions,
            **({"cabinet_plan": self.cabinet_plan,
                "cabinet_dimensions": self.cabinet_dimensions,
                "card_catalog": self.card_catalog} if include_config else {}),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
