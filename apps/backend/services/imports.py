import hashlib
import os
from typing import Iterable, List, Tuple

import pandas as pd
from sqlalchemy import select
from werkzeug.datastructures import FileStorage
from sqlalchemy.orm import Session

from apps.backend.db.models import (
    UploadedFile,
    UploadedFileType,
    IOListRow,
    Run,
    Project,
)
from apps.backend.utils import file_naming


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_upload_to_disk(file_storage: FileStorage, dest_dir: str, prefix: str = "") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = file_storage.filename or "upload.bin"
    if prefix:
        safe_name = f"{prefix}_{safe_name}"
    dest_path = os.path.join(dest_dir, safe_name)
    file_storage.save(dest_path)
    return dest_path


def store_uploaded_file(
    session: Session,
    file_storage: FileStorage,
    project: Project,
    run: Run,
    file_type: UploadedFileType,
    base_output_dir: str,
    save_bytes: bool = False,
) -> UploadedFile:
    project_dir = file_naming.get_project_output_dir(project.project_name)
    dest_dir = os.path.join(project_dir, "inputs")
    stored_path = _save_upload_to_disk(file_storage, dest_dir, prefix="raw")
    file_hash = compute_sha256(stored_path)

    # dedup within project/type
    existing = session.scalars(
        select(UploadedFile).where(
            UploadedFile.project_id == project.id,
            UploadedFile.file_type == file_type,
            UploadedFile.file_hash == file_hash,
        )
    ).first()
    if existing:
        existing.run_id = run.id
        session.flush()
        return existing

    uf = UploadedFile(
        project_id=project.id,
        run_id=run.id,
        file_type=file_type,
        original_name=file_storage.filename or "",
        stored_name=os.path.basename(stored_path),
        storage_path=stored_path,
        file_hash=file_hash,
        size_bytes=os.path.getsize(stored_path),
        mime_type=file_storage.mimetype,
        content=open(stored_path, "rb").read() if save_bytes else None,
    )
    session.add(uf)
    session.flush()
    return uf


def ingest_io_list_excel(
    session: Session, file_storage: FileStorage, project: Project, run: Run, base_output_dir: str
) -> Tuple[UploadedFile, List[IOListRow]]:
    uf = store_uploaded_file(session, file_storage, project, run, UploadedFileType.EXCEL, base_output_dir)
    df = pd.read_excel(uf.storage_path)
    rows: List[IOListRow] = []
    for idx, row in df.fillna("").iterrows():
        raw = row.to_dict()
        io_row = IOListRow(
            run_id=run.id,
            project_id=project.id,
            row_index=int(idx),
            jb=str(raw.get("JB") or raw.get("jb") or ""),
            io_type=str(raw.get("I/O Type") or raw.get("IO_TYPE") or ""),
            safety=str(raw.get("IS/NIS") or raw.get("SAFETY") or ""),
            location=str(raw.get("Location") or raw.get("LOCATION") or ""),
            terminal1=str(raw.get("terminal-1") or raw.get("TERM1") or ""),
            terminal2=str(raw.get("terminal-2") or raw.get("TERM2") or ""),
            src=str(raw.get("SRC") or ""),
            raw_json=raw,
        )
        rows.append(io_row)
        session.add(io_row)
    session.flush()
    return uf, rows


def ingest_pdf_files(
    session: Session,
    pdf_files: Iterable[FileStorage],
    project: Project,
    run: Run,
    base_output_dir: str,
    save_bytes: bool = False,
) -> List[UploadedFile]:
    stored: List[UploadedFile] = []
    for pdf in pdf_files:
        uf = store_uploaded_file(session, pdf, project, run, UploadedFileType.PDF, base_output_dir, save_bytes=save_bytes)
        stored.append(uf)
    return stored
