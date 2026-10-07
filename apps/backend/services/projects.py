import hashlib
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.backend.db.models import Project, Run


def _hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def get_or_create_project(
    session: Session,
    project_name: str,
    project_hash: Optional[str] = None,
    encoded_name: Optional[str] = None,
    reuse: bool = True,
) -> Project:
    """
    Idempotent project creation with name/hash reuse.
    """
    if not project_hash:
        project_hash = _hash_name(project_name)

    query = select(Project).where(
        (Project.project_name == project_name) | (Project.project_hash == project_hash)
    )
    project = session.scalar(query)
    if project:
        return project

    project = Project(
        project_name=project_name,
        project_hash=project_hash,
        encoded_name=encoded_name,
    )
    session.add(project)
    session.flush()
    return project


def latest_run(session: Session, project_id) -> Optional[Run]:
    stmt = (
        select(Run)
        .where(Run.project_id == project_id)
        .order_by(Run.created_at.desc())
    )
    return session.scalars(stmt).first()
