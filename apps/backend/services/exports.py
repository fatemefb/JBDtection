import os
from typing import Optional

from sqlalchemy.orm import Session

from apps.backend.db.models import ExportArtifact, ArtifactType, Run


def register_artifact(
    session: Session,
    run: Run,
    artifact_type: ArtifactType,
    path: str,
    size_bytes: Optional[int] = None,
    mime_type: Optional[str] = None,
    meta: Optional[dict] = None,
) -> ExportArtifact:
    art = (
        session.query(ExportArtifact)
        .filter_by(run_id=run.id, artifact_type=artifact_type)
        .one_or_none()
    )
    if art is None:
        art = ExportArtifact(
            run_id=run.id,
            project_id=run.project_id,
            artifact_type=artifact_type,
        )
        session.add(art)
    art.storage_path = path
    art.size_bytes = size_bytes or (os.path.getsize(path) if os.path.exists(path) else None)
    art.mime_type = mime_type
    art.ready_at = art.ready_at or None
    art.meta = meta
    session.flush()
    return art
