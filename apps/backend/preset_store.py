"""
preset_store.py — Self-contained storage for IO Assignment cabinet presets.

Stores per-project cabinet-type configurations (dimensions + card catalog +
cabinet_plan) so users can save a project's full cabinet setup, then reload
it later by selecting the preset from a dropdown.

Design choices:
  • Uses raw SQL via the project's existing SessionLocal — no ORM model
    registration needed, no migration scripts.
  • `CREATE TABLE IF NOT EXISTS` is called lazily on first use — safe to
    deploy without touching any existing files.
  • JSONB columns store the full DimensionMode state verbatim, so any
    future fields added to DimensionMode are automatically persisted.
  • Project name is unique — saving with an existing name updates that
    preset (matching the user's mental model of "save this project's
    settings").

Public API:
    list_presets()                  → list of {id, project_name, description, type_count, has_directions, updated_at}
    get_preset(preset_id)           → full preset row (including cabinet_dimensions JSON)
    get_preset_by_name(name)        → same, but lookup by project_name
    upsert_preset(payload)          → create or update by project_name, returns {id, action}
    delete_preset(preset_id)        → bool

Table schema:
    io_assignment_presets (
        id              SERIAL PRIMARY KEY,
        project_name    VARCHAR(255) NOT NULL UNIQUE,
        description     TEXT,
        type_count      INTEGER NOT NULL,
        has_directions  BOOLEAN NOT NULL,
        cabinet_plan    JSONB NOT NULL,         -- [{type, direction, quantity, priority}, ...]
        cabinet_dimensions JSONB NOT NULL,      -- {Type 1: {FRONT: {...}, REAR: {...}}, ...}
        card_catalog    JSONB NOT NULL,         -- {Barrier Board (AI): 300, ...}
        created_by      VARCHAR(100),
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        updated_at      TIMESTAMPTZ DEFAULT NOW()
    )
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Lazy session import ────────────────────────────────────────────────────
# We import SessionLocal lazily so this module can be loaded even if the
# host app's DB wiring isn't available (e.g. during unit testing).
_SessionLocal = None


def _get_session_factory():
    """Lazy accessor for the project's SessionLocal."""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    try:
        # Project layout uses apps.backend.db.session.SessionLocal
        from apps.backend.db.session import SessionLocal  # type: ignore
        _SessionLocal = SessionLocal
        return _SessionLocal
    except Exception:
        # Fallback: try alternative path used in some deployments
        try:
            from db.session import SessionLocal  # type: ignore
            _SessionLocal = SessionLocal
            return _SessionLocal
        except Exception as e:
            logger.error("preset_store: could not import SessionLocal: %s", e)
            raise


# ── Schema bootstrap ───────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS io_assignment_presets (
    id                SERIAL PRIMARY KEY,
    project_name      VARCHAR(255) NOT NULL UNIQUE,
    description       TEXT,
    type_count        INTEGER NOT NULL DEFAULT 1,
    has_directions    BOOLEAN NOT NULL DEFAULT TRUE,
    cabinet_plan      JSONB NOT NULL DEFAULT '[]'::jsonb,
    cabinet_dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
    card_catalog      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by        VARCHAR(100),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_io_assignment_presets_name
    ON io_assignment_presets (project_name);
"""

_schema_initialized = False


def _ensure_schema(db_session) -> None:
    """Idempotently create the preset table if it doesn't exist."""
    global _schema_initialized
    if _schema_initialized:
        return
    try:
        db_session.execute(_CREATE_TABLE_SQL)
        db_session.execute(_CREATE_INDEX_SQL)
        db_session.commit()
        _schema_initialized = True
        logger.info("preset_store: schema initialized (io_assignment_presets table ready)")
    except Exception as e:
        db_session.rollback()
        # Don't crash the request — log and continue. Read/write ops will then
        # raise a clearer error if the table truly doesn't exist.
        logger.error("preset_store: schema init failed: %s", e)


@contextmanager
def _db():
    """Yield a SQLAlchemy session bound to the project's DB."""
    SessionLocal = _get_session_factory()
    session = SessionLocal()
    try:
        _ensure_schema(session)
        yield session
    finally:
        session.close()


# ── Row serialization ──────────────────────────────────────────────────────
def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a SQLAlchemy Row to a plain dict (JSON-friendly)."""
    if row is None:
        return None
    cols = row._mapping.keys() if hasattr(row, "_mapping") else row.keys()
    out = {}
    for k in cols:
        v = getattr(row, k, None)
        if isinstance(v, datetime):
            v = v.isoformat()
        out[k] = v
    return out


def _row_summary(row) -> Dict[str, Any]:
    """Lightweight preset listing (no big JSON payloads)."""
    d = _row_to_dict(row)
    if d is None:
        return None
    return {
        "id": d.get("id"),
        "project_name": d.get("project_name"),
        "description": d.get("description"),
        "type_count": d.get("type_count"),
        "has_directions": d.get("has_directions"),
        "created_by": d.get("created_by"),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
    }


# ── Public API ─────────────────────────────────────────────────────────────
def list_presets() -> List[Dict[str, Any]]:
    """Return all presets as lightweight summaries, newest first."""
    from sqlalchemy import text
    with _db() as s:
        rows = s.execute(text(
            "SELECT id, project_name, description, type_count, has_directions, "
            "created_by, created_at, updated_at "
            "FROM io_assignment_presets ORDER BY updated_at DESC;"
        )).fetchall()
        return [_row_summary(r) for r in rows]


def get_preset(preset_id: int) -> Optional[Dict[str, Any]]:
    """Return one preset by id, including the full JSON state."""
    from sqlalchemy import text
    with _db() as s:
        row = s.execute(text(
            "SELECT * FROM io_assignment_presets WHERE id = :id;"
        ), {"id": int(preset_id)}).fetchone()
        return _row_to_dict(row)


def get_preset_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return one preset by project_name, including the full JSON state."""
    from sqlalchemy import text
    with _db() as s:
        row = s.execute(text(
            "SELECT * FROM io_assignment_presets WHERE project_name = :name;"
        ), {"name": str(name)}).fetchone()
        return _row_to_dict(row)


def upsert_preset(payload: Dict[str, Any], username: Optional[str] = None) -> Dict[str, Any]:
    """
    Create or update a preset by project_name.

    Required payload fields:
        - project_name (str)
        - type_count (int)
        - has_directions (bool)
        - cabinet_plan (list)
        - cabinet_dimensions (dict)
        - card_catalog (dict)
    Optional:
        - description (str)

    Returns: {"id": int, "action": "created"|"updated"}
    """
    from sqlalchemy import text

    project_name = str(payload.get("project_name", "")).strip()
    if not project_name:
        raise ValueError("project_name is required")

    type_count = int(payload.get("type_count", 1))
    has_directions = bool(payload.get("has_directions", True))
    description = payload.get("description", "") or ""
    cabinet_plan = payload.get("cabinet_plan", []) or []
    cabinet_dimensions = payload.get("cabinet_dimensions", {}) or {}
    card_catalog = payload.get("card_catalog", {}) or {}

    # Serialize JSON fields — SQLAlchemy text() accepts dict/list directly for
    # JSONB columns on psycopg2, but we normalize to JSON strings for safety
    # across drivers.
    cabinet_plan_json = json.dumps(cabinet_plan, ensure_ascii=False)
    cabinet_dimensions_json = json.dumps(cabinet_dimensions, ensure_ascii=False)
    card_catalog_json = json.dumps(card_catalog, ensure_ascii=False)

    with _db() as s:
        existing = s.execute(text(
            "SELECT id FROM io_assignment_presets WHERE project_name = :name;"
        ), {"name": project_name}).fetchone()

        if existing:
            s.execute(text(
                """
                UPDATE io_assignment_presets
                   SET description       = :description,
                       type_count        = :type_count,
                       has_directions    = :has_directions,
                       cabinet_plan      = CAST(:cabinet_plan AS jsonb),
                       cabinet_dimensions = CAST(:cabinet_dimensions AS jsonb),
                       card_catalog      = CAST(:card_catalog AS jsonb),
                       created_by        = COALESCE(:created_by, created_by),
                       updated_at        = NOW()
                 WHERE id = :id;
                """
            ), {
                "id": int(existing.id),
                "description": description,
                "type_count": type_count,
                "has_directions": has_directions,
                "cabinet_plan": cabinet_plan_json,
                "cabinet_dimensions": cabinet_dimensions_json,
                "card_catalog": card_catalog_json,
                "created_by": username,
            })
            s.commit()
            logger.info("preset_store: updated preset id=%s name=%r", existing.id, project_name)
            return {"id": int(existing.id), "action": "updated", "project_name": project_name}
        else:
            new_row = s.execute(text(
                """
                INSERT INTO io_assignment_presets
                    (project_name, description, type_count, has_directions,
                     cabinet_plan, cabinet_dimensions, card_catalog, created_by)
                VALUES
                    (:project_name, :description, :type_count, :has_directions,
                     CAST(:cabinet_plan AS jsonb),
                     CAST(:cabinet_dimensions AS jsonb),
                     CAST(:card_catalog AS jsonb),
                     :created_by)
                RETURNING id;
                """
            ), {
                "project_name": project_name,
                "description": description,
                "type_count": type_count,
                "has_directions": has_directions,
                "cabinet_plan": cabinet_plan_json,
                "cabinet_dimensions": cabinet_dimensions_json,
                "card_catalog": card_catalog_json,
                "created_by": username,
            }).fetchone()
            s.commit()
            new_id = int(new_row.id) if new_row else None
            logger.info("preset_store: created preset id=%s name=%r", new_id, project_name)
            return {"id": new_id, "action": "created", "project_name": project_name}


def delete_preset(preset_id: int) -> bool:
    """Delete a preset by id. Returns True if a row was actually deleted."""
    from sqlalchemy import text
    with _db() as s:
        result = s.execute(text(
            "DELETE FROM io_assignment_presets WHERE id = :id;"
        ), {"id": int(preset_id)})
        s.commit()
        deleted = (result.rowcount or 0) > 0
        if deleted:
            logger.info("preset_store: deleted preset id=%s", preset_id)
        return deleted
