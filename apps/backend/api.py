import os
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd
from flask import Blueprint, jsonify, request, send_file
from sqlalchemy import select
import sqlalchemy as sa

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


def _project_or_404(session, project_id) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise ValueError("project_not_found")
    return project


def _run_or_404(session, run_id) -> Run:
    run = session.get(Run, run_id)
    if not run:
        raise ValueError("run_not_found")
    return run


@api_bp.errorhandler(Exception)
def handle_exception(exc):
    code = 400
    if isinstance(exc, ValueError) and str(exc) in {"project_not_found", "run_not_found"}:
        code = 404
    return jsonify({"status": "error", "error": str(exc)}), code


@api_bp.route("/projects", methods=["POST"])
def create_project():
    payload = request.get_json(force=True)
    name = payload.get("project_name")
    if not name:
        return jsonify({"status": "error", "message": "project_name required"}), 400
    project_hash = payload.get("project_hash")
    encoded = payload.get("encoded_name")
    with session_scope() as session:
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
    with session_scope() as session:
        projects_rows = session.scalars(select(Project).order_by(Project.created_at.desc())).all()
        data = []
        for p in projects_rows:
            data.append(
                {
                    "id": str(p.id),
                    "name": p.project_name,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "last_finalized_run_id": str(p.last_finalized_run_id) if p.last_finalized_run_id else None,
                }
            )
        return jsonify({"status": "success", "projects": data})


@api_bp.route("/projects/<uuid:project_id>", methods=["GET"])
def get_project(project_id):
    with session_scope() as session:
        project = _project_or_404(session, project_id)
        latest = project_svc.latest_run(session, project.id)
        return jsonify(
            {
                "status": "success",
                "project": {
                    "id": str(project.id),
                    "name": project.project_name,
                    "hash": project.project_hash,
                    "created_at": project.created_at.isoformat() if project.created_at else None,
                    "last_finalized_run_id": str(project.last_finalized_run_id) if project.last_finalized_run_id else None,
                    "latest_run_id": str(latest.id) if latest else None,
                    "latest_run_status": latest.status.value if latest else None,
                },
            }
        )


@api_bp.route("/projects/<uuid:project_id>/runs", methods=["POST"])
def create_run(project_id):
    payload = request.get_json(force=True, silent=True) or {}
    reuse_existing = payload.get("reuse_existing", True)
    with session_scope() as session:
        project = _project_or_404(session, project_id)
        latest = project_svc.latest_run(session, project.id)
        if reuse_existing and latest and latest.status in {RunStatus.PENDING, RunStatus.PROCESSING, RunStatus.REVIEW}:
            run = latest
        else:
            run = run_svc.create_run(session, project, initiated_by=payload.get("initiated_by"))
            run_svc.keep_only_latest(session, project.id)
        session.commit()
        return jsonify({"status": "success", "run_id": str(run.id), "run_status": run.status.value})


@api_bp.route("/projects/<uuid:project_id>/runs/latest", methods=["GET"])
def get_latest_run(project_id):
    with session_scope() as session:
        project = _project_or_404(session, project_id)
        run = project_svc.latest_run(session, project.id)
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
    with session_scope() as session:
        run = _run_or_404(session, run_id)
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
    excel_file = request.files.get("file") or request.files.get("excel_file")
    if not excel_file:
        return jsonify({"status": "error", "message": "excel file required"}), 400
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        project = _project_or_404(session, run.project_id)
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
    pdfs = request.files.getlist("files") or request.files.getlist("pdf_files")
    if not pdfs:
        return jsonify({"status": "error", "message": "no pdfs provided"}), 400
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        project = _project_or_404(session, run.project_id)
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
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        _ = _project_or_404(session, run.project_id)
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

IO_MAPPED_KEYS = {
    "jb": ["JB", "JB No", "jb"],
    "io_type": ["I/O Type", "io_type"],
    "safety": ["IS/NIS", "safety"],
    "location": ["Location", "location"],
    "terminal1": ["terminal-1", "terminal1", "Terminal_First_Number"],
    "terminal2": ["terminal-2", "terminal2", "Terminal_Second_Number"],
    "src": ["SRC", "src"],
    "match_status": ["Match", "Match_Type", "match_status"],
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _payload_to_row_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    embedded = payload.get("values")
    if isinstance(embedded, dict):
        values.update(embedded)
    for key, val in payload.items():
        if key in {"values", "row_index"}:
            continue
        values[key] = val
    return values


def _sync_row_mapped_fields(row: IOListRow, raw_values: Dict[str, Any]) -> None:
    for field, aliases in IO_MAPPED_KEYS.items():
        value = None
        if field in raw_values:
            value = raw_values.get(field)
        else:
            for alias in aliases:
                if alias in raw_values:
                    value = raw_values.get(alias)
                    break
        if value is None:
            continue
        setattr(row, field, _normalize_text(value))
        for alias in aliases:
            raw_values[alias] = _normalize_text(value)


@api_bp.route("/runs/<uuid:run_id>/io-list", methods=["GET"])
def get_io_list(run_id):
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        rows = session.scalars(
            select(IOListRow).where(IOListRow.run_id == run.id).order_by(IOListRow.row_index)
        ).all()
        # determine available columns from raw_json
        columns = set(DEFAULT_IO_COLUMNS)
        for r in rows:
            if r.raw_json and isinstance(r.raw_json, dict):
                columns.update(r.raw_json.keys())
        columns = [c for c in DEFAULT_IO_COLUMNS] + [c for c in sorted(columns) if c not in DEFAULT_IO_COLUMNS]

        payload_rows = []
        for r in rows:
            values = {col: (r.raw_json or {}).get(col, "") for col in columns}
            # Keep mapped values in sync with editable fields so UI/export always reflect latest edits.
            mapped_values = {
                "jb": r.jb,
                "io_type": r.io_type,
                "safety": r.safety,
                "location": r.location,
                "terminal1": r.terminal1,
                "terminal2": r.terminal2,
                "src": r.src,
                "match_status": r.match_status,
            }
            for field, aliases in IO_MAPPED_KEYS.items():
                mapped = mapped_values.get(field)
                if mapped is None:
                    continue
                for alias in aliases:
                    values[alias] = mapped
            payload_rows.append(
                {
                    "id": str(r.id),
                    "row_index": r.row_index,
                    "jb": r.jb,
                    "io_type": r.io_type,
                    "safety": r.safety,
                    "location": r.location,
                    "terminal1": r.terminal1,
                    "terminal2": r.terminal2,
                    "src": r.src,
                    "normalized_tag": r.normalized_tag,
                    "match_status": r.match_status,
                    "issue_count": r.issue_count,
                    "values": values,
                }
            )
        return jsonify({"status": "success", "columns": columns, "rows": payload_rows})


@api_bp.route("/runs/<uuid:run_id>/io-list", methods=["POST"])
def add_io_row(run_id):
    payload = request.get_json(force=True, silent=True) or {}
    row_values = _payload_to_row_values(payload)
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        project = _project_or_404(session, run.project_id)
        next_idx = (
            session.scalar(select(sa.func.max(IOListRow.row_index)).where(IOListRow.run_id == run.id)) or 0
        ) + 1
        row_index = _safe_int(payload.get("row_index", row_values.get("row_index", next_idx)), next_idx)
        exists_same_index = session.scalar(
            select(sa.func.count(IOListRow.id)).where(IOListRow.run_id == run.id, IOListRow.row_index == row_index)
        )
        if exists_same_index:
            row_index = next_idx
        row = IOListRow(
            run_id=run.id,
            project_id=project.id,
            row_index=row_index,
            raw_json=row_values,
        )
        _sync_row_mapped_fields(row, row_values)
        row.normalized_tag = _normalize_text(
            row_values.get("normalized_tag")
            or row_values.get("Tag/SPARE")
            or row_values.get("Tag No")
            or row_values.get("tag")
        )
        if "issue_count" in row_values:
            row.issue_count = _safe_int(row_values.get("issue_count"), default=0)
        row.raw_json = row_values
        session.add(row)
        session.flush()
        return jsonify({"status": "success", "row_id": str(row.id), "row_index": row.row_index})


@api_bp.route("/io-list-rows/<uuid:row_id>", methods=["PATCH"])
def patch_io_row(row_id):
    payload = request.get_json(force=True, silent=True) or {}
    with session_scope() as session:
        row = session.get(IOListRow, row_id)
        if not row:
            return jsonify({"status": "error", "message": "row not found"}), 404

        raw = dict(row.raw_json) if isinstance(row.raw_json, dict) else {}
        incoming = _payload_to_row_values(payload)
        raw.update(incoming)
        _sync_row_mapped_fields(row, raw)

        if "normalized_tag" in incoming:
            row.normalized_tag = _normalize_text(incoming.get("normalized_tag"))
        elif "Tag/SPARE" in incoming:
            row.normalized_tag = _normalize_text(incoming.get("Tag/SPARE"))
        elif "Tag No" in incoming:
            row.normalized_tag = _normalize_text(incoming.get("Tag No"))

        if "issue_count" in incoming:
            row.issue_count = _safe_int(incoming.get("issue_count"), default=row.issue_count or 0)

        row.raw_json = raw
        session.flush()
        return jsonify({"status": "success", "row_id": str(row.id)})


@api_bp.route("/io-list-rows/<uuid:row_id>", methods=["DELETE"])
def delete_io_row(row_id):
    with session_scope() as session:
        row = session.get(IOListRow, row_id)
        if not row:
            return jsonify({"status": "error", "message": "row not found"}), 404
        session.delete(row)
        session.flush()
        return jsonify({"status": "success"})


@api_bp.route("/runs/<uuid:run_id>/issues", methods=["GET"])
def list_issues(run_id):
    severity_filter = request.args.get("severity")
    status_filter = request.args.get("status")
    with session_scope() as session:
        _run_or_404(session, run_id)
        stmt = select(Issue).where(Issue.run_id == run_id)
        if severity_filter:
            stmt = stmt.where(Issue.severity == IssueSeverity(severity_filter))
        if status_filter:
            stmt = stmt.where(Issue.status == IssueStatus(status_filter))
        issues = session.scalars(stmt.order_by(Issue.created_at)).all()
        payload = []
        for i in issues:
            payload.append(
                {
                    "id": str(i.id),
                    "severity": i.severity.value,
                    "status": i.status.value,
                    "message": i.message,
                    "code": i.code,
                    "io_list_row_id": str(i.io_list_row_id) if i.io_list_row_id else None,
                    "tag_occurrence_id": str(i.tag_occurrence_id) if i.tag_occurrence_id else None,
                }
            )
        return jsonify({"status": "success", "issues": payload})


@api_bp.route("/issues/<uuid:issue_id>", methods=["PATCH"])
def patch_issue(issue_id):
    payload = request.get_json(force=True)
    with session_scope() as session:
        issue = session.get(Issue, issue_id)
        if not issue:
            return jsonify({"status": "error", "message": "issue not found"}), 404
        if "status" in payload:
            issue.status = IssueStatus(payload["status"])
            if issue.status == IssueStatus.RESOLVED:
                issue.resolved_at = issue.resolved_at or issue.created_at
        session.flush()
        return jsonify({"status": "success", "issue_id": str(issue.id), "new_status": issue.status.value})


@api_bp.route("/runs/<uuid:run_id>/logs", methods=["GET"])
def get_run_logs(run_id):
    after = request.args.get("after")
    after_id = int(after) if after else None
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        logs = run_svc.get_logs(session, run.id, after_id=after_id, limit=800)
        payload = [
            {"id": l.id, "level": l.level, "message": l.message, "created_at": l.created_at.isoformat()} for l in logs
        ]
        return jsonify({"status": "success", "logs": payload})


@api_bp.route("/runs/<uuid:run_id>/export", methods=["POST"])
def export_run(run_id):
    """
    Build Excel export from edited io_list_rows (full visible columns + mapped fields).
    """
    with session_scope() as session:
        run = _run_or_404(session, run_id)
        project = _project_or_404(session, run.project_id)
        rows = session.scalars(select(IOListRow).where(IOListRow.run_id == run.id).order_by(IOListRow.row_index)).all()

        columns = set(DEFAULT_IO_COLUMNS)
        for row in rows:
            if row.raw_json and isinstance(row.raw_json, dict):
                columns.update(row.raw_json.keys())
        columns = [c for c in DEFAULT_IO_COLUMNS] + [c for c in sorted(columns) if c not in DEFAULT_IO_COLUMNS]

        payload_rows = []
        for row in rows:
            values = {col: (row.raw_json or {}).get(col, "") for col in columns}
            mapped_values = {
                "jb": row.jb,
                "io_type": row.io_type,
                "safety": row.safety,
                "location": row.location,
                "terminal1": row.terminal1,
                "terminal2": row.terminal2,
                "src": row.src,
                "match_status": row.match_status,
            }
            for field, aliases in IO_MAPPED_KEYS.items():
                mapped = mapped_values.get(field)
                if mapped is None:
                    continue
                for alias in aliases:
                    values[alias] = mapped
            payload_rows.append(values)

        df = pd.DataFrame(payload_rows, columns=columns)
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
    with session_scope() as session:
        art = session.get(ExportArtifact, export_id)
        if not art or not art.storage_path:
            return jsonify({"status": "error", "message": "artifact not found"}), 404
        if not os.path.exists(art.storage_path):
            return jsonify({"status": "error", "message": "file missing on disk"}), 404
        return send_file(art.storage_path, as_attachment=True)
