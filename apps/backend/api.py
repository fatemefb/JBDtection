import os
import glob
import re
import json
import threading
import time
import logging
from typing import Any, Dict, Optional
from datetime import datetime

import pandas as pd
from flask import Blueprint, jsonify, request, send_file, session as flask_session
from sqlalchemy import select
from sqlalchemy.exc import DataError, IntegrityError
import sqlalchemy as sacd

from apps.backend.db.session import get_session, session_scope
from apps.backend.db.models import (
    Project,
    Run,
    RunStatus,
    IOListRow,
    Issue,
    IssueStatus,
    IssueSeverity,
    ExportArtifact,
    ArtifactType,
    UploadedFileType,
)
from apps.backend.services import projects as project_svc
from apps.backend.services import runs as run_svc
from apps.backend.services import imports as import_svc
from apps.backend.services.exports import register_artifact
from apps.backend.utils import file_naming

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)


def _current_username() -> str:
    username = flask_session.get("username")
    if not username:
        raise PermissionError("unauthorized")
    return str(username)


def _is_admin(username: Optional[str]) -> bool:
    return str(username or "").strip().lower() == "admin"


def _project_or_404(session, project_id, username: Optional[str] = None, require_user_run: bool = True) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise ValueError("project_not_found")
    if username and not _is_admin(username):
        has_user_run = session.scalar(
            select(Run.id)
            .where(Run.project_id == project.id, Run.initiated_by == username)
            .limit(1)
        )
        if has_user_run:
            return project
        if require_user_run:
            raise ValueError("project_not_found")
        has_any_run = session.scalar(
            select(Run.id)
            .where(Run.project_id == project.id)
            .limit(1)
        )
        if has_any_run:
            raise ValueError("project_not_found")
    return project


def _run_or_404(session, run_id, username: Optional[str] = None) -> Run:
    run = session.get(Run, run_id)
    if not run:
        raise ValueError("run_not_found")
    if username and not _is_admin(username):
        if str(run.initiated_by or "").strip() != username:
            raise ValueError("run_not_found")
    return run


def _io_row_or_404(session, row_id, username: Optional[str] = None) -> IOListRow:
    row = session.get(IOListRow, row_id)
    if not row:
        raise ValueError("row_not_found")
    _run_or_404(session, row.run_id, username=username)
    return row


def _issue_or_404(session, issue_id, username: Optional[str] = None) -> Issue:
    issue = session.get(Issue, issue_id)
    if not issue:
        raise ValueError("issue_not_found")
    _run_or_404(session, issue.run_id, username=username)
    return issue


def _artifact_or_404(session, export_id, username: Optional[str] = None) -> ExportArtifact:
    art = session.get(ExportArtifact, export_id)
    if not art:
        raise ValueError("artifact_not_found")
    _run_or_404(session, art.run_id, username=username)
    return art


@api_bp.errorhandler(Exception)
def handle_exception(exc):
    code = 500
    if isinstance(exc, PermissionError) and str(exc) == "unauthorized":
        code = 401
    elif isinstance(exc, ValueError) and str(exc) in {
        "project_not_found",
        "run_not_found",
        "row_not_found",
        "issue_not_found",
        "artifact_not_found",
    }:
        code = 404
    else:
        logger.exception("Unhandled API exception", exc_info=exc)
    return jsonify({"status": "error", "error": str(exc)}), code


@api_bp.route("/status", methods=["GET"])
def api_status():
    return jsonify({"status": "ok"}), 200


@api_bp.route("/projects", methods=["POST"])
def create_project():
    username = _current_username()
    payload = request.get_json(force=True)
    name = payload.get("project_name")
    if not name:
        return jsonify({"status": "error", "message": "project_name required"}), 400
    project_hash = payload.get("project_hash")
    encoded = payload.get("encoded_name")
    with session_scope() as session:
        if not _is_admin(username):
            existing = session.scalar(select(Project).where(Project.project_name == name))
            if existing:
                user_has_access = session.scalar(
                    select(Run.id)
                    .where(Run.project_id == existing.id, Run.initiated_by == username)
                    .limit(1)
                )
                if not user_has_access:
                    return jsonify(
                        {
                            "status": "error",
                            "message": "project_name already exists and belongs to another user",
                        }
                    ), 409
        project = project_svc.get_or_create_project(session, name, project_hash, encoded, reuse=True)
        session.commit()
        return jsonify(
            {
                "status": "success",
                "project": {
                    "id": str(project.id),
                    "project_name": project.project_name,
                    "project_hash": project.project_hash,
                },
            }
        )


@api_bp.route("/projects", methods=["GET"])
def list_projects():
    username = _current_username()
    with session_scope() as session:
        if _is_admin(username):
            stmt = select(Project).order_by(Project.created_at.desc())
        else:
            stmt = (
                select(Project)
                .join(Run, Run.project_id == Project.id)
                .where(Run.initiated_by == username)
                .distinct()
                .order_by(Project.created_at.desc())
            )
        projects_rows = session.scalars(stmt).all()
        data = []
        for p in projects_rows:
            if _is_admin(username):
                last_finalized_run_id = str(p.last_finalized_run_id) if p.last_finalized_run_id else None
            else:
                latest_user_finalized = session.scalar(
                    select(Run.id)
                    .where(
                        Run.project_id == p.id,
                        Run.initiated_by == username,
                        Run.status == RunStatus.FINALIZED,
                    )
                    .order_by(Run.finished_at.desc(), Run.created_at.desc())
                )
                last_finalized_run_id = str(latest_user_finalized) if latest_user_finalized else None
            data.append(
                {
                    "id": str(p.id),
                    "name": p.project_name,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "last_finalized_run_id": last_finalized_run_id,
                }
            )
        return jsonify({"status": "success", "projects": data})


@api_bp.route("/projects/<uuid:project_id>", methods=["GET"])
def get_project(project_id):
    username = _current_username()
    with session_scope() as session:
        project = _project_or_404(session, project_id, username=username, require_user_run=False)
        if _is_admin(username):
            latest = project_svc.latest_run(session, project.id)
        else:
            latest = session.scalar(
                select(Run)
                .where(Run.project_id == project.id, Run.initiated_by == username)
                .order_by(Run.created_at.desc())
            )
        return jsonify(
            {
                "status": "success",
                "project": {
                    "id": str(project.id),
                    "name": project.project_name,
                    "hash": project.project_hash,
                    "created_at": project.created_at.isoformat() if project.created_at else None,
                    "last_finalized_run_id": str(latest.id) if latest else None,
                    "latest_run_id": str(latest.id) if latest else None,
                    "latest_run_status": latest.status.value if latest else None,
                },
            }
        )


@api_bp.route("/projects/<uuid:project_id>/runs", methods=["POST"])
def create_run(project_id):
    username = _current_username()
    payload = request.get_json(force=True, silent=True) or {}
    reuse_existing = payload.get("reuse_existing", True)
    with session_scope() as session:
        project = _project_or_404(session, project_id, username=username, require_user_run=False)
        if _is_admin(username):
            latest = project_svc.latest_run(session, project.id)
        else:
            latest = session.scalar(
                select(Run)
                .where(Run.project_id == project.id, Run.initiated_by == username)
                .order_by(Run.created_at.desc())
            )
        if reuse_existing and latest and latest.status in {RunStatus.PENDING, RunStatus.PROCESSING, RunStatus.REVIEW}:
            run = latest
        else:
            run = run_svc.create_run(session, project, initiated_by=username)
            run_svc.keep_only_latest(session, project.id, initiated_by=username)
        session.commit()
        return jsonify({"status": "success", "run_id": str(run.id), "run_status": run.status.value})


@api_bp.route("/projects/<uuid:project_id>/runs/latest", methods=["GET"])
def get_latest_run(project_id):
    username = _current_username()
    with session_scope() as session:
        project = _project_or_404(session, project_id, username=username)
        if _is_admin(username):
            run = project_svc.latest_run(session, project.id)
        else:
            run = session.scalar(
                select(Run)
                .where(Run.project_id == project.id, Run.initiated_by == username)
                .order_by(Run.created_at.desc())
            )
        if not run:
            return jsonify({"status": "error", "message": "no runs"}), 404
        return jsonify(
            {
                "status": "success",
                "run": {
                    "id": str(run.id),
                    "status": run.status.value,
                    "stage": run.stage,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                },
            }
        )


@api_bp.route("/runs/<uuid:run_id>/status", methods=["GET"])
def get_run_status(run_id):
    username = _current_username()
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        return jsonify(
            {
                "status": "success",
                "run": {
                    "id": str(run.id),
                    "project_id": str(run.project_id),
                    "status": run.status.value,
                    "stage": run.stage,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    "notes": run.notes,
                },
            }
        )


@api_bp.route("/runs/<uuid:run_id>/upload-io-list-excel", methods=["POST"])
def upload_io_list(run_id):
    username = _current_username()
    excel_file = request.files.get("file") or request.files.get("excel_file")
    if not excel_file:
        return jsonify({"status": "error", "message": "excel file required"}), 400
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        project = _project_or_404(session, run.project_id, username=username)
        uf, rows = import_svc.ingest_io_list_excel(session, excel_file, project, run, file_naming.BASE_OUTPUT_DIR)
        run_svc.set_status(session, run, RunStatus.REVIEW, stage="io_import")
        session.commit()
        return jsonify(
            {
                "status": "success",
                "uploaded_file_id": str(uf.id),
                "row_count": len(rows),
            }
        )


@api_bp.route("/runs/<uuid:run_id>/upload-pdfs", methods=["POST"])
def upload_pdfs(run_id):
    username = _current_username()
    pdfs = request.files.getlist("files") or request.files.getlist("pdf_files")
    if not pdfs:
        return jsonify({"status": "error", "message": "no pdfs provided"}), 400
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        project = _project_or_404(session, run.project_id, username=username)
        stored = import_svc.ingest_pdf_files(session, pdfs, project, run, file_naming.BASE_OUTPUT_DIR)
        session.commit()
        return jsonify(
            {"status": "success", "files": [{"id": str(f.id), "name": f.original_name} for f in stored]}
        )


def _simulate_processing(run_id):
    with session_scope() as session:
        run = session.get(Run, run_id)
        if not run:
            return
        run_svc.set_status(session, run, RunStatus.PROCESSING, stage="jbdetection")
        run_svc.add_log_line(session, run, "JBDetection started", level="info")
        session.commit()
        # simulate processing for now; wire actual extractor later
        for i in range(3):
            time.sleep(0.5)
            run_svc.add_log_line(session, run, f"processing chunk {i+1}/3", level="info")
            session.commit()
        run_svc.set_status(session, run, RunStatus.FINALIZED, stage="finalized")
        run_svc.add_log_line(session, run, "JBDetection finished (simulated)", level="info")
        session.commit()


@api_bp.route("/runs/<uuid:run_id>/start-jbdetection", methods=["POST"])
def start_jbdetection(run_id):
    """
    Kicks off processing; currently simulated and writes logs into DB.
    """
    username = _current_username()
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        _ = _project_or_404(session, run.project_id, username=username)
        # spawn background thread
        t = threading.Thread(target=_simulate_processing, args=(run.id,), daemon=True)
        t.start()
        return jsonify({"status": "accepted", "run_id": str(run.id)})


DEFAULT_IO_COLUMNS = [
    "Loop No",
    "Tag No",
    "Description",
    "P&ID No",
    "Unit No",
    "JB No",
    "JB Terminal",
    "Multi Cable",
    "Pair/Core",
    "Service",
    "Location",
    "ITR",
    "I/O Type",
    "Signal Type",
    "Signal Con",
    "Contact T",
    "IS/NIS",
    "terminal-1",
    "terminal-2",
    "SRC",
]


def _normalize_row_values(raw_json: Any) -> Dict[str, Any]:
    if not isinstance(raw_json, dict):
        return {}
    nested = raw_json.get("values")
    if isinstance(nested, dict):
        values = dict(nested)
        for key, value in raw_json.items():
            if key in {"values", "row_index"}:
                continue
            if key not in values:
                values[key] = value
        return values
    return dict(raw_json)


def _first_present(values: Dict[str, Any], aliases) -> Any:
    for key in aliases:
        if key in values:
            return values.get(key)
    return None


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(str(value).strip())
    except Exception:
        return default


def _load_run_pattern_config(run_id: str) -> Dict[str, Any]:
    """
    Best-effort load of pattern config used in processing from task files.
    """
    tasks_dir = os.path.join(file_naming.BASE_OUTPUT_DIR, ".tasks")
    if not os.path.isdir(tasks_dir):
        return {}

    candidate_files = sorted(
        glob.glob(os.path.join(tasks_dir, "*.json")),
        key=lambda p: os.path.getmtime(p),
        reverse=True
    )
    for task_path in candidate_files:
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except Exception:
            continue

        if str(task_data.get("run_id", "")).strip() != str(run_id).strip():
            continue

        result = task_data.get("result") if isinstance(task_data.get("result"), dict) else {}
        patterns = result.get("patterns_used") if isinstance(result.get("patterns_used"), dict) else {}
        if patterns:
            return patterns

    return {}


def _evaluate_x_expression(pattern_text: str, x_value: int) -> str:
    def replace_expr(match):
        expr = match.group(1)
        expr = expr.replace("x", str(int(x_value)))
        try:
            result = eval(expr)
            return str(int(result))
        except Exception:
            return match.group(0)

    return re.sub(r"\{([^}]+)\}", replace_expr, str(pattern_text or ""))


def _generate_terminals_for_index(tag_number: int, pattern_cfg: Dict[str, Any]) -> Dict[str, str]:
    pattern = str(pattern_cfg.get("terminal_pattern") or "").strip()
    include_scr = bool(pattern_cfg.get("include_scr", True))

    if not pattern:
        return {
            "terminal_first": str(tag_number),
            "terminal_second": str(tag_number + 1),
            "scr_terminal": "",
            "full_string": f"{tag_number}, {tag_number + 1}",
        }

    rendered = _evaluate_x_expression(pattern, int(tag_number))
    if not include_scr:
        rendered = re.sub(r",?\s*SCR\s*,?", "", rendered, flags=re.IGNORECASE)
        rendered = re.sub(r",\s*,", ",", rendered).strip(", ")

    parts = [p.strip() for p in rendered.split(",") if str(p).strip()]
    scr_terminal = ""
    non_scr_parts = []
    for p in parts:
        if "SCR" in p.upper() and not scr_terminal:
            scr_terminal = p
        else:
            non_scr_parts.append(p)

    terminal_first = non_scr_parts[0] if len(non_scr_parts) > 0 else ""
    terminal_second = non_scr_parts[1] if len(non_scr_parts) > 1 else ""

    return {
        "terminal_first": terminal_first,
        "terminal_second": terminal_second,
        "scr_terminal": scr_terminal,
        "full_string": rendered,
    }


def _generate_wire_for_index(tag_number: int, pattern_cfg: Dict[str, Any]) -> Dict[str, str]:
    pattern = str(pattern_cfg.get("wire_color_pattern") or "").strip()
    if not pattern:
        first = f"BK{int(tag_number):02d}"
        second = f"WT{int(tag_number):02d}"
        return {"wire_text": f"{first}, {second}", "wire_1": first, "wire_2": second}

    def replace_number(match):
        width_group = match.group(1)
        if width_group:
            try:
                width = int(width_group)
            except Exception:
                width = 0
            if width > 0:
                return str(int(tag_number)).zfill(width)
        return str(int(tag_number))

    rendered = re.sub(r"\{x(?::0?(\d+)d)?\}", replace_number, pattern, flags=re.IGNORECASE)
    parts = [p.strip() for p in rendered.split(",") if str(p).strip()]
    return {
        "wire_text": rendered,
        "wire_1": parts[0] if len(parts) > 0 else "",
        "wire_2": parts[1] if len(parts) > 1 else "",
    }


def _normalize_jb_text(jb_value: Any) -> str:
    return str(jb_value or "").strip().upper()


def _resolve_issue_annotated_pdf_path(session, issue: Issue) -> Optional[str]:
    details = issue.details if isinstance(issue.details, dict) else {}
    pdf_name = os.path.basename(str(details.get("pdf_name") or "").strip())
    if not pdf_name:
        return None
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name = f"{pdf_name}.pdf"
    wanted_names = {
        pdf_name.lower(),
        f"annotated_{pdf_name}".lower(),
    }

    run = _run_or_404(session, issue.run_id)
    project = _project_or_404(session, run.project_id, require_user_run=False)
    safe_project_name = re.sub(r"[^\w\-]", "_", str(project.project_name or "").strip())
    project_dir = os.path.join(file_naming.BASE_OUTPUT_DIR, safe_project_name)
    annotated_dir = os.path.join(project_dir, "annotated_pdfs")

    candidate_paths = []
    if os.path.isdir(annotated_dir):
        for ext in ("*.pdf", "*.PDF"):
            candidate_paths.extend(glob.glob(os.path.join(annotated_dir, ext)))

    if run.artifacts:
        for art in run.artifacts:
            if art.artifact_type == ArtifactType.ANNOTATED_PDF and art.storage_path:
                candidate_paths.append(art.storage_path)

    # 1) exact filename match
    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        base = os.path.basename(path).lower()
        if base in wanted_names:
            return path

    # 2) fuzzy fallback by original pdf stem
    wanted_stem = os.path.splitext(pdf_name)[0].lower()
    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        base = os.path.basename(path).lower()
        if wanted_stem and wanted_stem in base:
            return path

    return None


def _set_row_values(row: IOListRow, merged_values: Dict[str, Any]) -> None:
    row.raw_json = merged_values
    row.jb = str(_first_present(merged_values, ["JB", "JB No", "jb"]) or row.jb or "").strip()
    row.terminal1 = str(_first_present(merged_values, ["terminal-1", "Terminal_First_Number", "terminal1"]) or row.terminal1 or "").strip()
    row.terminal2 = str(_first_present(merged_values, ["terminal-2", "Terminal_Second_Number", "terminal2"]) or row.terminal2 or "").strip()
    row.src = str(_first_present(merged_values, ["SRC", "src"]) or row.src or "").strip()
    row.match_status = str(_first_present(merged_values, ["Match", "match_status"]) or row.match_status or "").strip()


def _infer_insert_row_index_for_candidate(session, run_id, jb_text: str, candidate_tag_number: Optional[int]) -> Optional[int]:
    if not jb_text or not candidate_tag_number or candidate_tag_number <= 0:
        return None

    jb_rows = session.scalars(
        select(IOListRow)
        .where(IOListRow.run_id == run_id)
        .order_by(IOListRow.row_index)
    ).all()
    jb_rows = [r for r in jb_rows if _normalize_jb_text(r.jb) == _normalize_jb_text(jb_text)]
    if not jb_rows:
        return None

    target_pos = max(1, int(candidate_tag_number))
    if target_pos <= len(jb_rows):
        return int(jb_rows[target_pos - 1].row_index)
    return int(jb_rows[-1].row_index) + 1


def _reindex_jb_rows(session, run: Run, jb_text: str, pattern_cfg: Dict[str, Any]) -> int:
    normalized_target_jb = _normalize_jb_text(jb_text)
    if not normalized_target_jb:
        return 0

    rows = session.scalars(
        select(IOListRow)
        .where(IOListRow.run_id == run.id)
        .order_by(IOListRow.row_index)
    ).all()
    jb_rows = [r for r in rows if _normalize_jb_text(r.jb) == normalized_target_jb]
    if not jb_rows:
        return 0

    changed = 0
    for idx, row in enumerate(jb_rows, start=1):
        merged_values = _normalize_row_values(row.raw_json)
        terminals = _generate_terminals_for_index(idx, pattern_cfg)
        wires = _generate_wire_for_index(idx, pattern_cfg)

        merged_values["Tag_Number"] = idx
        merged_values["terminal-1"] = terminals["terminal_first"]
        merged_values["terminal-2"] = terminals["terminal_second"]
        merged_values["Terminal_First_Number"] = terminals["terminal_first"]
        merged_values["Terminal_Second_Number"] = terminals["terminal_second"]
        merged_values["SCR_Terminal_Number"] = terminals["scr_terminal"]
        merged_values["Wire_Code_1"] = wires["wire_1"]
        merged_values["Wire_Code_2"] = wires["wire_2"]
        merged_values["Wire Colors"] = wires["wire_text"]
        merged_values["Tag_Number_Status"] = "Reindexed by JB order"

        _set_row_values(row, merged_values)
        changed += 1

    session.flush()
    return changed


def _shift_io_row_indices(session, run_id, start_index: int, delta: int = 1) -> int:
    """
    Shift row_index for a run without violating uq_io_row_idx.

    A bulk UPDATE like `row_index = row_index + 1 WHERE row_index >= X` can violate the
    unique constraint mid-statement (e.g. 2 -> 3 while 3 still exists). We instead lock
    and update in a safe order.
    """
    if not start_index or start_index <= 0 or not delta:
        return 0

    # Fast + safe strategy: move affected rows to a high offset region (no collisions),
    # then move them back with the desired delta.
    max_idx = session.scalar(
        select(sacd.func.max(IOListRow.row_index)).where(IOListRow.run_id == run_id)
    ) or 0
    offset = int(max_idx) + abs(int(delta)) + 1000

    affected = session.scalar(
        select(sacd.func.count(IOListRow.id)).where(IOListRow.run_id == run_id, IOListRow.row_index >= int(start_index))
    ) or 0
    if not affected:
        return 0

    # Phase 1: jump to high numbers
    session.execute(
        sacd.update(IOListRow)
        .where(IOListRow.run_id == run_id, IOListRow.row_index >= int(start_index))
        .values(row_index=IOListRow.row_index + int(offset))
    )
    session.flush()

    # Phase 2: apply delta while returning to original band
    session.execute(
        sacd.update(IOListRow)
        .where(IOListRow.run_id == run_id, IOListRow.row_index >= int(start_index) + int(offset))
        .values(row_index=IOListRow.row_index - int(offset) + int(delta))
    )
    session.flush()
    return int(affected)


@api_bp.route("/runs/<uuid:run_id>/io-list", methods=["GET"])
def get_io_list(run_id):
    username = _current_username()
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        rows = session.scalars(
            select(IOListRow).where(IOListRow.run_id == run.id).order_by(IOListRow.row_index)
        ).all()
        # determine available columns from raw_json
        columns = set(DEFAULT_IO_COLUMNS)
        for r in rows:
            columns.update(_normalize_row_values(r.raw_json).keys())
        columns = [c for c in DEFAULT_IO_COLUMNS] + [c for c in sorted(columns) if c not in DEFAULT_IO_COLUMNS]

        payload_rows = []
        for r in rows:
            raw_values = _normalize_row_values(r.raw_json)
            values = {col: raw_values.get(col, "") for col in columns}
            # fallback to mapped fields
            if "JB No" in values and not values.get("JB No"):
                values["JB No"] = r.jb or ""
            if "I/O Type" in values and not values.get("I/O Type"):
                values["I/O Type"] = r.io_type or ""
            if "IS/NIS" in values and not values.get("IS/NIS"):
                values["IS/NIS"] = r.safety or ""
            if "Location" in values and not values.get("Location"):
                values["Location"] = r.location or ""
            if "terminal-1" in values and not values.get("terminal-1"):
                values["terminal-1"] = r.terminal1 or ""
            if "terminal-2" in values and not values.get("terminal-2"):
                values["terminal-2"] = r.terminal2 or ""
            if "SRC" in values and not values.get("SRC"):
                values["SRC"] = r.src or ""
            if "Match" in values and not values.get("Match"):
                values["Match"] = r.match_status or ""
            payload_rows.append(
                {
                    "id": str(r.id),
                    "row_index": r.row_index,
                    "values": values,
                }
            )
        return jsonify({"status": "success", "columns": columns, "rows": payload_rows})


@api_bp.route("/runs/<uuid:run_id>/io-list", methods=["POST"])
def add_io_row(run_id):
    username = _current_username()
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        payload = {}
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        project = _project_or_404(session, run.project_id, username=username)
        next_idx = (
            session.scalar(select(sacd.func.max(IOListRow.row_index)).where(IOListRow.run_id == run.id)) or 0
        ) + 1
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        if not values:
            values = {k: v for k, v in payload.items() if k not in {"row_index", "values"}}
        jb = payload.get("jb")
        if jb is None:
            jb = _first_present(values, ["JB", "JB No", "jb"])
        io_type = payload.get("io_type")
        if io_type is None:
            io_type = _first_present(values, ["I/O Type", "io_type"])
        safety = payload.get("safety")
        if safety is None:
            safety = _first_present(values, ["IS/NIS", "safety"])
        location = payload.get("location")
        if location is None:
            location = _first_present(values, ["Location", "location"])
        terminal1 = payload.get("terminal1")
        if terminal1 is None:
            terminal1 = _first_present(values, ["terminal-1", "Terminal_First_Number", "terminal1"])
        terminal2 = payload.get("terminal2")
        if terminal2 is None:
            terminal2 = _first_present(values, ["terminal-2", "Terminal_Second_Number", "terminal2"])
        src = payload.get("src")
        if src is None:
            src = _first_present(values, ["SRC", "src"])
        match_status = payload.get("match_status")
        if match_status is None:
            match_status = _first_present(values, ["Match", "match_status"])

        row = IOListRow(
            run_id=run.id,
            project_id=project.id,
            row_index=payload.get("row_index", next_idx),
            jb=jb,
            io_type=io_type,
            safety=safety,
            location=location,
            terminal1=terminal1,
            terminal2=terminal2,
            src=src,
            match_status=match_status,
            raw_json=values,
        )
        session.add(row)
        session.flush()
        return jsonify({"status": "success", "row_id": str(row.id), "row_index": row.row_index})


@api_bp.route("/io-list-rows/<uuid:row_id>", methods=["PATCH"])
def patch_io_row(row_id):
    username = _current_username()
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        payload = {}
    with session_scope() as session:
        row = _io_row_or_404(session, row_id, username=username)
        if "row_index" in payload:
            try:
                row.row_index = int(payload.get("row_index"))
            except Exception:
                pass
        values_update = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        merged_values = _normalize_row_values(row.raw_json)
        merged_values.update(values_update)
        if values_update:
            row.raw_json = merged_values

        # explicit top-level fields keep priority
        for field in ["jb", "io_type", "safety", "location", "terminal1", "terminal2", "src", "match_status"]:
            if field in payload:
                setattr(row, field, payload[field])

        # infer canonical fields from values payload when present
        if values_update:
            if any(k in values_update for k in ["JB", "JB No", "jb"]):
                row.jb = _first_present(merged_values, ["JB", "JB No", "jb"])
            if any(k in values_update for k in ["I/O Type", "io_type"]):
                row.io_type = _first_present(merged_values, ["I/O Type", "io_type"])
            if any(k in values_update for k in ["IS/NIS", "safety"]):
                row.safety = _first_present(merged_values, ["IS/NIS", "safety"])
            if any(k in values_update for k in ["Location", "location"]):
                row.location = _first_present(merged_values, ["Location", "location"])
            if any(k in values_update for k in ["terminal-1", "Terminal_First_Number", "terminal1"]):
                row.terminal1 = _first_present(merged_values, ["terminal-1", "Terminal_First_Number", "terminal1"])
            if any(k in values_update for k in ["terminal-2", "Terminal_Second_Number", "terminal2"]):
                row.terminal2 = _first_present(merged_values, ["terminal-2", "Terminal_Second_Number", "terminal2"])
            if any(k in values_update for k in ["SRC", "src"]):
                row.src = _first_present(merged_values, ["SRC", "src"])
            if any(k in values_update for k in ["Match", "match_status"]):
                row.match_status = _first_present(merged_values, ["Match", "match_status"])
        session.flush()
        return jsonify({"status": "success", "row_id": str(row.id)})


@api_bp.route("/io-list-rows/<uuid:row_id>", methods=["DELETE"])
def delete_io_row(row_id):
    username = _current_username()
    with session_scope() as session:
        row = _io_row_or_404(session, row_id, username=username)
        session.delete(row)
        session.flush()
        return jsonify({"status": "success"})


@api_bp.route("/runs/<uuid:run_id>/issues", methods=["GET"])
def list_issues(run_id):
    username = _current_username()
    severity_filter = request.args.get("severity")
    status_filter = request.args.get("status")
    with session_scope() as session:
        _run_or_404(session, run_id, username=username)
        stmt = select(Issue).where(Issue.run_id == run_id)
        if severity_filter:
            stmt = stmt.where(Issue.severity == IssueSeverity(severity_filter))
        if status_filter:
            stmt = stmt.where(Issue.status == IssueStatus(status_filter))
        issues = session.scalars(stmt.order_by(Issue.created_at)).all()
        payload = []
        for i in issues:
            details = i.details if isinstance(i.details, dict) else {}
            linked_row_index = i.io_row.row_index if i.io_row else None
            has_pdf_context = bool(str(details.get("pdf_name") or "").strip())
            annotated_preview_url = f"/api/issues/{i.id}/annotated-pdf" if has_pdf_context else None
            payload.append(
                {
                    "id": str(i.id),
                    "severity": i.severity.value,
                    "status": i.status.value,
                    "message": i.message,
                    "code": i.code,
                    "io_list_row_id": str(i.io_list_row_id) if i.io_list_row_id else None,
                    "tag_occurrence_id": str(i.tag_occurrence_id) if i.tag_occurrence_id else None,
                    "details": details,
                    "pdf_name": details.get("pdf_name"),
                    "page": details.get("page"),
                    "jb": details.get("jb"),
                    "mc": details.get("mc"),
                    "ocr_text": details.get("ocr_text") or details.get("display_text"),
                    "score": details.get("score"),
                    "tag_number": details.get("tag_number"),
                    "terminal_first_number": details.get("terminal_first_number"),
                    "terminal_second_number": details.get("terminal_second_number"),
                    "wire_code_1": details.get("wire_code_1"),
                    "wire_code_2": details.get("wire_code_2"),
                    "annotated_preview_url": annotated_preview_url,
                    "can_add_to_io": i.code == "unmatched_pattern_candidate" and not i.io_list_row_id,
                    "linked_row_index": linked_row_index,
                }
            )
        return jsonify({"status": "success", "issues": payload})


@api_bp.route("/issues/<uuid:issue_id>", methods=["PATCH"])
def patch_issue(issue_id):
    username = _current_username()
    payload = request.get_json(force=True)
    with session_scope() as session:
        issue = _issue_or_404(session, issue_id, username=username)
        if "status" in payload:
            issue.status = IssueStatus(payload["status"])
            if issue.status == IssueStatus.RESOLVED:
                issue.resolved_at = issue.resolved_at or issue.created_at
        session.flush()
        return jsonify({"status": "success", "issue_id": str(issue.id), "new_status": issue.status.value})


@api_bp.route("/issues/<uuid:issue_id>/annotated-pdf", methods=["GET"])
def get_issue_annotated_pdf(issue_id):
    username = _current_username()
    with session_scope() as session:
        issue = _issue_or_404(session, issue_id, username=username)
        pdf_path = _resolve_issue_annotated_pdf_path(session, issue)
        if not pdf_path:
            return jsonify({"status": "error", "message": "annotated pdf not found for issue"}), 404
        if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
            return jsonify({"status": "error", "message": "annotated pdf path is invalid"}), 404
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False)


@api_bp.route("/issues/<uuid:issue_id>/add-to-io", methods=["POST"])
def add_issue_to_io(issue_id):
    """
    Create an IO row from an image-only unmatched pattern issue, then link and resolve the issue.
    """
    username = _current_username()
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    logger.info(f"[add_issue_to_io] issue_id={issue_id} payload_keys={list(payload.keys()) if isinstance(payload, dict) else 'n/a'}")
    with session_scope() as session:
        issue = _issue_or_404(session, issue_id, username=username)
        if issue.code != "unmatched_pattern_candidate":
            return jsonify({"status": "error", "message": "issue is not addable to io list", "code": issue.code}), 400

        if issue.io_list_row_id:
            existing = session.get(IOListRow, issue.io_list_row_id)
            return jsonify(
                {
                    "status": "success",
                    "row_id": str(issue.io_list_row_id),
                    "row_index": existing.row_index if existing else None,
                    "already_exists": True,
                }
            )

        details = issue.details if isinstance(issue.details, dict) else {}
        values_payload = payload.get("values") if isinstance(payload.get("values"), dict) else {}

        def _pick_value(keys, fallback=None):
            for key in keys:
                if key in values_payload and str(values_payload.get(key)).strip() != "":
                    return str(values_payload.get(key)).strip()
            return fallback

        run = _run_or_404(session, issue.run_id, username=username)
        project = _project_or_404(session, issue.project_id, username=username)

        # Serialize modifications per-run to reduce concurrent index shift conflicts.
        session.execute(select(Run.id).where(Run.id == run.id).with_for_update())

        tag_text = _pick_value(["Tag No", "Tag/SPARE", "Tag", "tag", "tag_no"], None)
        if not tag_text:
            tag_text = str(details.get("ocr_text") or details.get("display_text") or "").strip()
        if not tag_text and issue.message:
            # Fallback: try to extract last token from message like "Pattern-like tag not in IO List: XXX"
            parts = str(issue.message).split(":")
            if len(parts) >= 2:
                tag_text = parts[-1].strip()
        if not tag_text:
            logger.warning(f"[add_issue_to_io] Missing tag_text. payload={payload} details={details}")
            return jsonify({"status": "error", "message": "issue has no OCR tag text", "debug": {"payload": payload, "details_keys": list(details.keys()) if isinstance(details, dict) else []}}), 400

        next_idx = (
            session.scalar(select(sacd.func.max(IOListRow.row_index)).where(IOListRow.run_id == run.id)) or 0
        ) + 1
        try:
            requested_row_index = int(payload.get("row_index")) if payload.get("row_index") is not None else None
        except Exception:
            requested_row_index = None
        prefer_inferred_index = str(payload.get("prefer_inferred_index", "true")).strip().lower() not in {"0", "false", "no"}
        candidate_tag_number = _safe_int(details.get("tag_number"), default=None)
        inferred_row_index = _infer_insert_row_index_for_candidate(
            session=session,
            run_id=run.id,
            jb_text=(payload.get("jb") or _pick_value(["JB", "JB No", "jb"]) or details.get("jb") or ""),
            candidate_tag_number=candidate_tag_number
        )
        if prefer_inferred_index and inferred_row_index and inferred_row_index > 0:
            row_index = int(inferred_row_index)
        elif requested_row_index and requested_row_index > 0:
            row_index = int(requested_row_index)
        elif inferred_row_index and inferred_row_index > 0:
            row_index = int(inferred_row_index)
        else:
            row_index = int(next_idx)

        def _fit(value, max_len):
            if value is None:
                return ""
            value = str(value).strip()
            return value[:max_len]

        pdf_name = _fit(_pick_value(["PDF_Name", "pdf_name"], details.get("pdf_name")) or "", 240)
        page = payload.get("page", details.get("page"))
        jb = _fit(payload.get("jb") or _pick_value(["JB", "JB No", "jb"]) or details.get("jb") or "", 128)
        mc = _fit(payload.get("mc") or _pick_value(["MC", "Multi Cable", "mc"]) or details.get("mc") or "", 128)
        score = payload.get("score", details.get("score"))
        reason = _fit(payload.get("reason") or _pick_value(["Reason", "Description"], details.get("reason")) or "", 2000)
        wire_colors_list = details.get("wire_colors") if isinstance(details.get("wire_colors"), list) else []
        if not wire_colors_list:
            wire_colors_list = details.get("raw_cable_descriptions") or details.get("cable_descriptions") or []
        terminal_first = _fit(details.get("terminal_first_number") or details.get("Terminal_First_Number") or "", 128)
        terminal_second = _fit(details.get("terminal_second_number") or details.get("Terminal_Second_Number") or "", 128)
        scr_terminal = _fit(details.get("scr_terminal_number") or details.get("SCR_Terminal_Number") or "", 128)
        tag_number = candidate_tag_number
        wire_code_1 = _fit(details.get("wire_code_1") or details.get("Wire_Code_1") or "", 128)
        wire_code_2 = _fit(details.get("wire_code_2") or details.get("Wire_Code_2") or "", 128)
        cable_code = _fit(details.get("cable_code") or details.get("Cable_Code") or "", 256)
        cable_description = _fit(details.get("cable_description") or details.get("Cable_Description") or "", 2000)
        tag_number_status = _fit(details.get("tag_number_status") or details.get("Tag_Number_Status") or "", 256)
        derived_type = _fit(details.get("type") or details.get("Type") or "Tag", 64)
        page_label = _fit(f"{pdf_name} - Page {page}" if pdf_name and page is not None else (pdf_name or ""), 128)
        src = payload.get("src") or _pick_value(["SRC", "Source"], "OCR Pattern Candidate")
        if not src:
            src = "OCR Pattern Candidate"
        if score is not None:
            try:
                src = f"{src} ({float(score):.2f})"
            except Exception:
                pass
        src = _fit(src, 256)

        requested_match_status = payload.get("match_status") or _pick_value(["Match"], "") or "Added from image candidate"
        match_status = _fit(requested_match_status, 32)

        row_raw = {**values_payload} if values_payload else {}
        # ensure core fields exist
        row_raw.setdefault("Tag No", tag_text)
        row_raw.setdefault("Tag/SPARE", tag_text)
        row_raw.setdefault("JB", jb)
        row_raw.setdefault("JB No", jb)
        row_raw.setdefault("MC", mc)
        row_raw.setdefault("Multi Cable", mc)
        if tag_number is not None and str(tag_number).strip() != "":
            row_raw.setdefault("Tag_Number", tag_number)
        row_raw.setdefault("Wire_Code_1", wire_code_1)
        row_raw.setdefault("Wire_Code_2", wire_code_2)
        row_raw.setdefault("Terminal_First_Number", terminal_first)
        row_raw.setdefault("Terminal_Second_Number", terminal_second)
        row_raw.setdefault("SCR_Terminal_Number", scr_terminal)
        row_raw.setdefault("Cable_Code", cable_code)
        row_raw.setdefault("Cable_Description", cable_description)
        row_raw.setdefault("Type", derived_type)
        row_raw.setdefault("Tag_Number_Status", tag_number_status)
        row_raw.setdefault("Location", page_label)
        row_raw.setdefault("Description", reason)
        row_raw.setdefault("SRC", src)
        row_raw.setdefault("Match", requested_match_status)
        row_raw.setdefault("PDF_Name", pdf_name)
        row_raw.setdefault("Page", page)
        row_raw.setdefault("OCR_Text", tag_text)
        row_raw.setdefault("Candidate_Score", score)
        row_raw.setdefault("Reason", reason)
        if not row_raw.get("Cable_Description"):
            row_raw.setdefault(
                "Cable_Description",
                ", ".join(details.get("raw_cable_descriptions") or details.get("cable_descriptions") or []),
            )
        if wire_colors_list:
            row_raw.setdefault("Wire Colors", ", ".join(wire_colors_list))
        if terminal_first or terminal_second:
            row_raw.setdefault("Terminals", ", ".join([v for v in [terminal_first, terminal_second] if v]))
        row_raw.setdefault("Source_Issue_Id", str(issue.id))

        # ایجاد فضا برای درج در index هدف (برای حفظ ترتیب واقعی) - شیفت امن برای جلوگیری از uq_io_row_idx
        try:
            _shift_io_row_indices(session, run.id, int(row_index), delta=1)
        except IntegrityError as exc:
            session.rollback()
            logger.warning(f"[add_issue_to_io] IntegrityError while shifting rows for run_id={run.id}: {exc}")
            return jsonify({"status": "error", "message": "row shift conflict (row_index)", "error": str(exc)}), 409

        io_row = IOListRow(
            run_id=run.id,
            project_id=project.id,
            row_index=row_index,
            jb=jb,
            io_type="",
            safety="",
            location=page_label,
            terminal1=terminal_first,
            terminal2=terminal_second,
            src=src,
            normalized_tag=_fit(tag_text, 256),
            match_status=match_status,
            raw_json=row_raw,
            issue_count=1,
        )
        try:
            session.add(io_row)
            session.flush()
        except IntegrityError as exc:
            session.rollback()
            logger.warning(f"[add_issue_to_io] IntegrityError for issue_id={issue_id}: {exc}")
            return jsonify({"status": "error", "message": "row insert conflict (row_index or relation)", "error": str(exc)}), 409
        except DataError as exc:
            session.rollback()
            logger.warning(f"[add_issue_to_io] DataError for issue_id={issue_id}: {exc}")
            return jsonify({"status": "error", "message": "invalid field size or format for database columns", "error": str(exc)}), 400

        issue.io_list_row_id = io_row.id
        issue.status = IssueStatus.RESOLVED
        issue.resolved_at = datetime.utcnow()
        issue.resolved_by = "ui_add_to_io"
        session.flush()

        pattern_cfg = _load_run_pattern_config(str(run.id))
        reindexed_count = _reindex_jb_rows(session, run, jb, pattern_cfg)

        return jsonify(
            {
                "status": "success",
                "row_id": str(io_row.id),
                "row_index": io_row.row_index,
                "issue_id": str(issue.id),
                "issue_status": issue.status.value,
                "insert_mode": "inferred_by_tag_number" if (prefer_inferred_index and inferred_row_index) else (
                    "requested" if requested_row_index else ("inferred_by_tag_number" if inferred_row_index else "append")
                ),
                "suggested_tag_number": candidate_tag_number,
                "effective_row_index": row_index,
                "reindexed_jb": jb,
                "reindexed_count": reindexed_count,
                "message": f"Candidate inserted at row {io_row.row_index}; {reindexed_count} row(s) in JB '{jb}' reindexed.",
            }
        )


@api_bp.route("/runs/<uuid:run_id>/logs", methods=["GET"])
def get_run_logs(run_id):
    username = _current_username()
    after = request.args.get("after")
    after_id = int(after) if after else None
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        logs = run_svc.get_logs(session, run.id, after_id=after_id, limit=800)
        payload = [
            {"id": l.id, "level": l.level, "message": l.message, "created_at": l.created_at.isoformat()} for l in logs
        ]
        return jsonify({"status": "success", "logs": payload})


@api_bp.route("/runs/<uuid:run_id>/export", methods=["POST"])
def export_run(run_id):
    """
    Build an Excel export from DB rows while preserving the run's original output column layout.
    """
    username = _current_username()
    with session_scope() as session:
        run = _run_or_404(session, run_id, username=username)
        project = _project_or_404(session, run.project_id, username=username)
        rows = session.scalars(select(IOListRow).where(IOListRow.run_id == run.id).order_by(IOListRow.row_index)).all()

        # 1) Input IO List columns (must stay first)
        input_columns = []
        try:
            excel_uploads = [
                f for f in (run.files or [])
                if f.file_type == UploadedFileType.EXCEL and f.storage_path and os.path.exists(f.storage_path)
            ]
            if excel_uploads:
                excel_uploads.sort(key=lambda f: f.created_at or datetime.min, reverse=True)
                input_df = pd.read_excel(excel_uploads[0].storage_path, nrows=1)
                input_columns = list(input_df.columns)
        except Exception as exc:
            logger.warning(f"Failed reading uploaded IO list columns for run {run.id}: {exc}")

        # 2) Process output columns (for derived/image columns order)
        final_columns = []
        final_art = session.scalar(
            select(ExportArtifact)
            .where(
                ExportArtifact.run_id == run.id,
                ExportArtifact.artifact_type == ArtifactType.FINAL_EXCEL,
                ExportArtifact.storage_path.isnot(None),
            )
            .order_by(ExportArtifact.created_at.desc())
        )
        if final_art and final_art.storage_path and os.path.exists(final_art.storage_path):
            try:
                template_df = pd.read_excel(final_art.storage_path, nrows=1)
                final_columns = list(template_df.columns)
            except Exception as exc:
                logger.warning(f"Failed reading template FINAL_EXCEL for run {run.id}: {exc}")

        # 3) Compose export columns: IO input first, then derived/output columns
        template_columns = []
        template_set = set()
        for col in input_columns:
            if col not in template_set:
                template_columns.append(col)
                template_set.add(col)
        for col in final_columns:
            if col not in template_set:
                template_columns.append(col)
                template_set.add(col)

        # 4) Ensure DB-added columns also appear (at the end)
        for r in rows:
            for col in _normalize_row_values(r.raw_json).keys():
                if col not in template_set:
                    template_columns.append(col)
                    template_set.add(col)

        # 5) Last fallback when no template found at all
        if not template_columns:
            for col in DEFAULT_IO_COLUMNS + ["Match"]:
                if col not in template_set:
                    template_columns.append(col)
                    template_set.add(col)

        exported_rows = []
        for r in rows:
            raw_values = _normalize_row_values(r.raw_json)
            out = {col: raw_values.get(col, "") for col in template_columns}

            # Canonical DB fields must be reflected even if raw_json is stale
            if "JB No" in out and (out["JB No"] is None or str(out["JB No"]).strip() == ""):
                out["JB No"] = r.jb or ""
            if "JB" in out and (out["JB"] is None or str(out["JB"]).strip() == ""):
                out["JB"] = r.jb or ""
            if "I/O Type" in out and (out["I/O Type"] is None or str(out["I/O Type"]).strip() == ""):
                out["I/O Type"] = r.io_type or ""
            if "IS/NIS" in out and (out["IS/NIS"] is None or str(out["IS/NIS"]).strip() == ""):
                out["IS/NIS"] = r.safety or ""
            if "Location" in out and (out["Location"] is None or str(out["Location"]).strip() == ""):
                out["Location"] = r.location or ""
            if "terminal-1" in out and (out["terminal-1"] is None or str(out["terminal-1"]).strip() == ""):
                out["terminal-1"] = r.terminal1 or ""
            if "terminal-2" in out and (out["terminal-2"] is None or str(out["terminal-2"]).strip() == ""):
                out["terminal-2"] = r.terminal2 or ""
            if "SRC" in out and (out["SRC"] is None or str(out["SRC"]).strip() == ""):
                out["SRC"] = r.src or ""
            if "Match" in out and (out["Match"] is None or str(out["Match"]).strip() == ""):
                out["Match"] = r.match_status or ""
            if "Tag No" in out and (out["Tag No"] is None or str(out["Tag No"]).strip() == ""):
                out["Tag No"] = r.normalized_tag or ""
            if "Tag/SPARE" in out and (out["Tag/SPARE"] is None or str(out["Tag/SPARE"]).strip() == ""):
                out["Tag/SPARE"] = r.normalized_tag or ""

            exported_rows.append(out)

        df = pd.DataFrame(exported_rows, columns=template_columns)
        project_dir = file_naming.get_project_output_dir(project.project_name)
        excel_name = file_naming.generate_document_filename(project.project_name, "ExcelFinal", "xlsx", directory=project_dir)
        excel_path = os.path.join(project_dir, excel_name)
        df.to_excel(excel_path, index=False)
        art = register_artifact(session, run, ArtifactType.FINAL_EXCEL, excel_path, mime_type="application/vnd.ms-excel")

        zip_path = file_naming.create_zip_archive(project.project_name, [excel_path], doc_type="Results")
        zip_art = register_artifact(session, run, ArtifactType.ZIP_BUNDLE, zip_path, mime_type="application/zip")
        session.commit()
        return jsonify(
            {
                "status": "success",
                "export_id": str(zip_art.id),
                "excel_path": excel_path,
                "zip_path": zip_path,
                "download_url": file_naming.get_download_url(zip_path),
            }
        )


@api_bp.route("/exports/<uuid:export_id>/download", methods=["GET"])
def download_export(export_id):
    username = _current_username()
    with session_scope() as session:
        art = _artifact_or_404(session, export_id, username=username)
        if not art.storage_path:
            return jsonify({"status": "error", "message": "artifact not found"}), 404
        if not os.path.exists(art.storage_path):
            return jsonify({"status": "error", "message": "file missing on disk"}), 404
        return send_file(art.storage_path, as_attachment=True)
