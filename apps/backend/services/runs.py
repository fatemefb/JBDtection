from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from apps.backend.db.models import Run, RunStatus, RunLog, Project


def create_run(session: Session, project: Project, initiated_by: str = None, reuse_of=None) -> Run:
    # force Enum value (lower-case) to avoid uppercase strings slipping in
    run = Run(
        project_id=project.id,
        status=RunStatus.PENDING,
        initiated_by=initiated_by,
        reuse_of_run_id=reuse_of,
    )
    session.add(run)
    session.flush()
    return run


def set_status(session: Session, run: Run, status: RunStatus, stage: str = None, notes: str = None):
    # accept both Enum and raw strings; normalize to lowercase Enum value
    if isinstance(status, str):
        status = RunStatus(status.lower())
    run.status = status
    if stage:
        run.stage = stage
    if status == RunStatus.PROCESSING and run.started_at is None:
        run.started_at = datetime.utcnow()
    if status in (RunStatus.FINALIZED, RunStatus.FAILED) and run.finished_at is None:
        run.finished_at = datetime.utcnow()
    if notes:
        run.notes = notes
    session.flush()


def add_log_line(session: Session, run: Run, message: str, level: str = "info"):
    entry = RunLog(
        run_id=run.id,
        project_id=run.project_id,
        message=message,
        level=level,
    )
    session.add(entry)
    session.flush()
    return entry


def get_logs(session: Session, run_id, after_id: Optional[int] = None, limit: int = 500) -> List[RunLog]:
    stmt = select(RunLog).where(RunLog.run_id == run_id)
    if after_id:
        stmt = stmt.where(RunLog.id > after_id)
    stmt = stmt.order_by(RunLog.id).limit(limit)
    return list(session.scalars(stmt))


def keep_only_latest(session: Session, project_id, initiated_by: str = None):
    """
    Delete/archive older runs if keep_latest_only flag is true.
    """
    stmt = select(Run.id).where(Run.project_id == project_id)
    if initiated_by is not None:
        stmt = stmt.where(Run.initiated_by == initiated_by)
    stmt = stmt.order_by(Run.created_at.desc()).offset(1)
    old_ids = [row for row in session.scalars(stmt)]
    if old_ids:
        update_stmt = (
            update(Run)
            .where(Run.id.in_(old_ids), Run.keep_latest_only.is_(True))
            .values(status=RunStatus.ARCHIVED, finished_at=datetime.utcnow())
        )
        if initiated_by is not None:
            update_stmt = update_stmt.where(Run.initiated_by == initiated_by)
        session.execute(update_stmt)
