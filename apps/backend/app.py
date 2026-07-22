from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from typing import List, Tuple, Dict, Set, Optional, Any, Union
import os
import re
import gc
import json
import math
import numpy as np
import pandas as pd
import cv2
import fitz 
import traceback
import tempfile
import platform
import shutil  
from pathlib import Path
import pytesseract
import time
import zipfile
from multiprocessing import Pool, cpu_count
import subprocess  
import tkinter as tk
import sys
import uuid
import threading
from datetime import datetime, timedelta
import copy
import fcntl
from sqlalchemy import select
# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tkinter import filedialog 
from logger_config import get_logger, LoggerMixin
from TagJBExtractorLogger import LoggedTagJBExtractor
from LinuxTagJBExtractorLogger import LoggedLinuxTagJBExtractor
from DataAnalysisModule import DataAnalysis, TagJBExtractor
from werkzeug.utils import secure_filename
from apps.backend.utils.file_naming import (
    BASE_OUTPUT_DIR,
    get_project_output_dir,
    get_log_dir,
    generate_document_filename,
    generate_log_filename,
    create_zip_archive,
    get_download_url,
)
from apps.backend.modules.io_assignment import run_io_assignment
from apps.backend.api import api_bp
from apps.backend.db.session import SessionLocal, session_scope
from apps.backend.services import projects as project_svc, runs as run_svc, imports as import_svc
from apps.backend.services.exports import register_artifact
from apps.backend.db.models import (
    ExportArtifact,
    ArtifactType,
    Run,
    RunStatus,
    Project,
    IOListRow,
    Issue,
    IssueSeverity,
    IssueStatus,
    UploadedFileType,
)
from apps.backend.modules.io_assignment.dimension_api import dimension_bp

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'frontend', 'static')
)
app.register_blueprint(api_bp)
app.register_blueprint(dimension_bp)

# تنظیم کلید محرمانه برای session
app.secret_key = 'jb_detection_system_secret_key'

# Configure upload folder for temporary files
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# اطمینان از وجود دایرکتوری پایه
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# تنظیم مسیر پیش‌فرض Tesseract بر اساس محیط اجرا
DEFAULT_TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "/usr/local/bin/tesseract")

os.environ["TESSDATA_PREFIX"] = "/usr/local/share/tessdata"
os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/bin"
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

PDF_CLASSIFIER_MODEL_PATH = os.environ.get(
    'PDF_CLASSIFIER_MODEL_PATH',
    os.path.join(BASE_DIR, 'modules', 'keras_model.h5')
)
PDF_CLASSIFIER_LABELS_PATH = os.environ.get(
    'PDF_CLASSIFIER_LABELS_PATH',
    os.path.join(BASE_DIR, 'modules', 'labels.txt')
)

# کاربران مجاز
VALID_USERS = {
    'admin': 'admin123',
    'user': 'user123',
    'cpec':'cpec@123'
}


# ایجاد لاگر برای فایل اصلی
logger = get_logger('app')

# دایرکتوری ذخیره task‌ها
TASKS_DIR = os.path.join(BASE_OUTPUT_DIR, '.tasks')
os.makedirs(TASKS_DIR, exist_ok=True)

# فایل قفل مشترک
LOCK_FILE = os.path.join(TASKS_DIR, '.lock')

class FileTaskManager:
    """مدیریت task‌ها با استفاده از فایل (thread-safe و worker-safe)"""
    
    @staticmethod
    def _get_task_file(task_id):
        """دریافت مسیر فایل task"""
        return os.path.join(TASKS_DIR, f"{task_id}.json")
    
    @staticmethod
    def _get_lock():
        """دریافت قفل سراسری با timeout"""
        lock_file = open(LOCK_FILE, 'w')
        # استفاده از LOCK_EX برای قفل انحصاری
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return lock_file
    
    @staticmethod
    def _release_lock(lock_file):
        """آزادسازی قفل"""
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
        except:
            pass
    
    @staticmethod
    def create_task(task_id, task_data):
        """ایجاد یک task جدید با atomic write"""
        lock_file = None
        try:
            lock_file = FileTaskManager._get_lock()
            
            task_data['created_at'] = datetime.now().isoformat()
            task_file = FileTaskManager._get_task_file(task_id)
            
            # نوشتن atomic با tempfile
            with tempfile.NamedTemporaryFile(
                mode='w', 
                delete=False, 
                dir=TASKS_DIR,
                suffix='.tmp',
                encoding='utf-8'
            ) as temp_file:
                json.dump(task_data, temp_file, indent=2, ensure_ascii=False)
                temp_name = temp_file.name
            
            # جایگزینی atomic
            os.replace(temp_name, task_file)
            logger.info(f"Task {task_id} created successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error creating task {task_id}: {e}")
            return False
        finally:
            if lock_file:
                FileTaskManager._release_lock(lock_file)
    
    @staticmethod
    def get_task(task_id):
        """دریافت یک task"""
        try:
            task_file = FileTaskManager._get_task_file(task_id)
            
            if not os.path.exists(task_file):
                return None
            
            with open(task_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None
    
    @staticmethod
    def update_task(task_id, updates):
        """به‌روزرسانی یک task با atomic write"""
        lock_file = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                lock_file = FileTaskManager._get_lock()
                task_file = FileTaskManager._get_task_file(task_id)
                
                if not os.path.exists(task_file):
                    logger.warning(f"Task {task_id} file not found")
                    return False
                
                # خواندن
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                # به‌روزرسانی
                task_data.update(updates)
                
                # نوشتن atomic
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    delete=False,
                    dir=TASKS_DIR,
                    suffix='.tmp',
                    encoding='utf-8'
                ) as temp_file:
                    json.dump(task_data, temp_file, indent=2, ensure_ascii=False)
                    temp_name = temp_file.name
                
                os.replace(temp_name, task_file)
                
                logger.info(f"Task {task_id} updated: status={updates.get('status', 'N/A')}, progress={updates.get('progress', 'N/A')}%")
                return True
                    
            except Exception as e:
                logger.error(f"Error updating task {task_id} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1)
            finally:
                if lock_file:
                    FileTaskManager._release_lock(lock_file)
        
        return False
    
    @staticmethod
    def delete_task(task_id):
        """حذف یک task"""
        lock_file = None
        try:
            lock_file = FileTaskManager._get_lock()
            task_file = FileTaskManager._get_task_file(task_id)
            
            if os.path.exists(task_file):
                os.remove(task_file)
                logger.info(f"Task {task_id} deleted")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            return False
        finally:
            if lock_file:
                FileTaskManager._release_lock(lock_file)
    
    @staticmethod
    def list_user_tasks(username):
        """لیست task‌های یک کاربر"""
        try:
            tasks = {}
            
            for filename in os.listdir(TASKS_DIR):
                if filename.endswith('.json') and not filename.startswith('.'):
                    task_id = filename[:-5]
                    task_data = FileTaskManager.get_task(task_id)
                    
                    if task_data and task_data.get('username') == username:
                        tasks[task_id] = {
                            'status': task_data.get('status'),
                            'progress': task_data.get('progress', 0),
                            'created_at': task_data.get('created_at'),
                            'project_name': task_data.get('project_name'),
                            'pdf_count': task_data.get('pdf_count', 0)
                        }
            
            return tasks
        except Exception as e:
            logger.error(f"Error listing tasks for {username}: {e}")
            return {}
    
    @staticmethod
    def cleanup_old_tasks():
        """حذف task‌های قدیمی‌تر از 24 ساعت"""
        try:
            current_time = datetime.now()
            deleted_count = 0
            
            for filename in os.listdir(TASKS_DIR):
                if filename.endswith('.json') and not filename.startswith('.'):
                    task_id = filename[:-5]
                    task_data = FileTaskManager.get_task(task_id)
                    
                    if task_data:
                        try:
                            created_at = datetime.fromisoformat(
                                task_data.get('created_at', current_time.isoformat())
                            )
                            age = current_time - created_at
                            
                            if age > timedelta(hours=24):
                                FileTaskManager.delete_task(task_id)
                                deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error processing task {task_id} for cleanup: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old task(s)")
                
        except Exception as e:
            logger.error(f"Error in cleanup_old_tasks: {e}")

# استفاده از FileTaskManager به جای دیکشنری TASKS
TaskManager = FileTaskManager


def _collect_user_task_file_paths(username: str) -> Set[str]:
    """Collect absolute output file paths from task payloads owned by the user."""
    collected: Set[str] = set()
    if not username:
        return collected

    def _walk(value):
        if isinstance(value, dict):
            for v in value.values():
                _walk(v)
            return
        if isinstance(value, list):
            for v in value:
                _walk(v)
            return
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return
            try:
                abs_candidate = os.path.abspath(candidate)
            except Exception:
                return
            if os.path.exists(abs_candidate):
                collected.add(abs_candidate)

    try:
        for filename in os.listdir(TASKS_DIR):
            if not filename.endswith('.json') or filename.startswith('.'):
                continue
            task_id = filename[:-5]
            task_data = FileTaskManager.get_task(task_id)
            if not task_data or task_data.get('username') != username:
                continue
            _walk(task_data.get('result'))
    except Exception:
        return collected
    return collected


class TaskStatus:
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STALE = 'stale'  # task whose worker died (timeout/OOM)

# ─────────────────────────────────────────────────────────────────
# HEARTBEAT MECHANISM — protects against worker TIMEOUT / OOM / SIGKILL
# ─────────────────────────────────────────────────────────────────
# When a gunicorn worker is killed (timeout/OOM), the processing thread
# dies but the task JSON stays in 'processing' status forever. The
# heartbeat mechanism solves this:
#   1. Worker thread updates `last_heartbeat` every N seconds
#   2. /task-status endpoint checks if heartbeat is stale (> 2 min)
#   3. Stale tasks are automatically marked as FAILED
#   4. Client receives proper error instead of hanging forever

HEARTBEAT_INTERVAL_SECONDS = 30          # update heartbeat every 30s (was 15 — less file lock contention)
HEARTBEAT_STALE_THRESHOLD_SECONDS = 1800  # 30 MINUTES — was 120s (too aggressive, caused false alarms during long OCR)
HEARTBEAT_CHECK_INTERVAL_SECONDS = 300   # scheduler checks every 5 min (was 60s — less overhead)

# Track active heartbeat threads so we can stop them on completion
_active_heartbeats: Dict[str, threading.Event] = {}
_heartbeats_lock = threading.Lock()

def start_heartbeat(task_id: str) -> threading.Event:
    """Start a background thread that updates task heartbeat every N seconds.
    Returns an Event that can be set to stop the heartbeat."""
    stop_event = threading.Event()
    with _heartbeats_lock:
        _active_heartbeats[task_id] = stop_event

    def _heartbeat_loop():
        while not stop_event.is_set():
            try:
                TaskManager.update_task(task_id, {
                    'last_heartbeat': datetime.now().isoformat()
                })
                logger.debug(f"Heartbeat tick for task {task_id}")
            except Exception as e:
                logger.warning(f"Heartbeat update failed for {task_id}: {e}")
            stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name=f"hb-{task_id[:8]}")
    hb_thread.start()
    logger.info(f"Heartbeat started for task {task_id}")
    return stop_event


def stop_heartbeat(task_id: str):
    """Stop the heartbeat thread for a task."""
    with _heartbeats_lock:
        stop_event = _active_heartbeats.pop(task_id, None)
    if stop_event:
        stop_event.set()
        logger.info(f"Heartbeat stopped for task {task_id}")


def is_task_stale(task_data: dict) -> bool:
    """Check if a task is stale (worker died, no heartbeat for a while)."""
    if not task_data:
        return False
    status = task_data.get('status')
    if status not in (TaskStatus.PROCESSING, TaskStatus.PENDING):
        return False
    last_hb = task_data.get('last_heartbeat')
    if not last_hb:
        started = task_data.get('started_at')
        if not started:
            return False
        try:
            started_dt = datetime.fromisoformat(started)
            age = (datetime.now() - started_dt).total_seconds()
            return age > 600
        except Exception:
            return False
    try:
        last_hb_dt = datetime.fromisoformat(last_hb)
        age = (datetime.now() - last_hb_dt).total_seconds()
        return age > HEARTBEAT_STALE_THRESHOLD_SECONDS
    except Exception:
        return False


def mark_stale_tasks_as_failed() -> int:
    """Scan all tasks and mark stale ones as failed.
    Returns count of tasks marked as stale."""
    marked = 0
    try:
        for filename in os.listdir(TASKS_DIR):
            if not filename.endswith('.json') or filename.startswith('.'):
                continue
            task_id = filename[:-5]
            task_data = FileTaskManager.get_task(task_id)
            if not task_data:
                continue
            if is_task_stale(task_data):
                logger.warning(f"Task {task_id} is stale — marking as failed")
                FileTaskManager.update_task(task_id, {
                    'status': TaskStatus.FAILED,
                    'error': f'Worker died (timeout/OOM/SIGKILL). No heartbeat for >{HEARTBEAT_STALE_THRESHOLD_SECONDS} seconds.',
                    'error_details': 'Task marked as stale by heartbeat monitor. The gunicorn worker was likely killed due to timeout or out-of-memory.',
                    'progress': 100,
                    'failed_at': datetime.now().isoformat(),
                    'failure_reason': 'worker_died'
                })
                stop_heartbeat(task_id)
                run_id = task_data.get('run_id')
                if run_id:
                    try:
                        with session_scope() as db:
                            run = db.get(Run, run_id)
                            if run and run.status not in (RunStatus.FAILED, RunStatus.FINALIZED):
                                run_svc.set_status(db, run, RunStatus.FAILED, stage="jbdetection",
                                                    notes="Worker died (timeout/OOM)")
                                run_svc.add_log_line(db, run,
                                    "Worker process killed (timeout/OOM). Task marked as stale.",
                                    level="error")
                                db.commit()
                    except Exception as db_err:
                        logger.error(f"Failed to update DB run {run_id} for stale task: {db_err}")
                marked += 1
    except Exception as e:
        logger.error(f"Error in mark_stale_tasks_as_failed: {e}")
    if marked > 0:
        logger.info(f"Marked {marked} stale task(s) as failed")
    return marked


def append_task_log(task_id, message):
    task = TaskManager.get_task(task_id) or {}
    logs = task.get('log', [])
    logs.append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })
    TaskManager.update_task(task_id, {'log': logs})

def to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)

def get_platform_specific_extractor(tesseract_path=None, excel_path=None):
    """
    بر اساس سیستم عامل، کلاس مناسب استخراج کننده را برمی‌گرداند
    """
    system = platform.system().lower()
    
    if system == 'linux':
        try:
            logger.info("استفاده از استخراج کننده مخصوص لینوکس با پشتیبانی از GPU و قابلیت لاگینگ")
            return LoggedLinuxTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری LoggedLinuxTagJBExtractor: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
       
    elif system == 'windows':
        try:
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ در ویندوز")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده ویندوز: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    elif system == 'darwin':  # macOS
        try:
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ در macOS")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده macOS: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    else:
        logger.info(f"سیستم عامل ناشناخته '{system}'، استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
        return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)


def _normalize_project_name_for_lookup(value: str) -> str:
    return re.sub(r'\s+', ' ', (value or '').strip()).lower()


def _is_admin_username(username: Optional[str]) -> bool:
    return str(username or "").strip().lower() == "admin"


def get_latest_excel_from_db(project_id: Optional[str] = None, project_name: Optional[str] = None,
                             username: Optional[str] = None) -> Optional[str]:
    """Return path to latest finalized run excel artifact by project name/id."""
    session = SessionLocal()
    try:
        current_user = str(username or "").strip()
        user_filter_active = bool(current_user) and not _is_admin_username(current_user)
        run = None
        if project_name:
            normalized_query = _normalize_project_name_for_lookup(project_name)
            if not normalized_query:
                return None
            if user_filter_active:
                projects = session.scalars(
                    select(Project)
                    .join(Run, Run.project_id == Project.id)
                    .where(Run.initiated_by == current_user)
                    .distinct()
                ).all()
            else:
                projects = session.scalars(select(Project)).all()
            exact_matches = [
                project for project in projects
                if _normalize_project_name_for_lookup(project.project_name) == normalized_query
            ]
            candidates = exact_matches or [
                project for project in projects
                if normalized_query in _normalize_project_name_for_lookup(project.project_name)
            ]

            latest_candidate = None
            latest_ts = None
            for candidate in candidates:
                stmt = (
                    select(Run)
                    .where(Run.project_id == candidate.id, Run.status == RunStatus.FINALIZED)
                    .order_by(Run.finished_at.desc(), Run.created_at.desc())
                )
                if user_filter_active:
                    stmt = stmt.where(Run.initiated_by == current_user)
                candidate_run = session.scalar(stmt)
                if not candidate_run:
                    continue
                run_ts = candidate_run.finished_at or candidate_run.started_at or candidate_run.created_at
                if latest_candidate is None or (run_ts and (latest_ts is None or run_ts > latest_ts)):
                    latest_candidate = candidate_run
                    latest_ts = run_ts
            run = latest_candidate
        elif project_id:
            stmt = (
                select(Run)
                .where(Run.project_id == project_id, Run.status == RunStatus.FINALIZED)
                .order_by(Run.finished_at.desc(), Run.created_at.desc())
            )
            if user_filter_active:
                stmt = stmt.where(Run.initiated_by == current_user)
            run = session.scalar(stmt)

        if not run:
            return None
        art = session.scalar(
            select(ExportArtifact)
            .where(
                ExportArtifact.run_id == run.id,
                ExportArtifact.artifact_type == ArtifactType.FINAL_EXCEL
            )
            .order_by(ExportArtifact.created_at.desc())
        )
        return art.storage_path if art and art.storage_path and os.path.exists(art.storage_path) else None
    finally:
        session.close()

def get_io_assignment_logger(project_name: str, username: str):
    safe_project_name = re.sub(r'[^\w\-]', '_', project_name)
    logger_name = f"io_assignment_{safe_project_name}"
    return get_logger(logger_name, username=username, project_name=project_name)

def _persist_run_outputs(run_id, project_id, output_excel_path, unmatched_excel_path, report_path, zip_path,
                         annotated_pdfs, unmatched_excel_tags, unmatched_pdf_tags,
                         pattern_unmatched_details=None, pattern_unmatched_candidates=None):
    """Persist extractor outputs (rows, artifacts, issues) into the relational DB."""
    with session_scope() as db:
        run = db.get(Run, run_id)
        project = db.get(Project, project_id) if project_id else (run.project if run else None)
        if not run or not project:
            return

        # replace any existing rows for this run
        db.query(IOListRow).filter(IOListRow.run_id == run.id).delete()

        if output_excel_path and os.path.exists(output_excel_path):
            df = pd.read_excel(output_excel_path)
            spare_col = "JB_SPARE_COUNT"
            if not df.empty and spare_col not in df.columns:
                if "JB" in df.columns:
                    upper_tags = df.get("Tag/SPARE", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
                    type_col = df.get("Type", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
                    spare_mask = type_col.eq("SPARE") | upper_tags.str.contains("SPARE", na=False)
                    jb_series = df["JB"].astype(str).str.strip()
                    spare_counts_by_jb = (
                        df.loc[spare_mask]
                        .assign(_JB_NORM=jb_series[spare_mask].str.upper())
                        .groupby("_JB_NORM")
                        .size()
                        .to_dict()
                    )

                    def _jb_spare_count(jb_value):
                        key = str(jb_value or "").strip().upper()
                        return int(spare_counts_by_jb.get(key, 0))

                    df[spare_col] = df["JB"].apply(_jb_spare_count)
                else:
                    df[spare_col] = 0
                df.to_excel(output_excel_path, index=False)

            for idx, row in df.fillna("").iterrows():
                raw = row.to_dict()
                db.add(
                    IOListRow(
                        run_id=run.id,
                        project_id=project.id,
                        row_index=int(idx) + 1,
                        jb=str(raw.get("JB") or raw.get("jb") or ""),
                        io_type=str(raw.get("I/O Type") or raw.get("IO_TYPE") or raw.get("io_type") or ""),
                        safety=str(raw.get("IS/NIS") or raw.get("SAFETY") or raw.get("is/nis") or ""),
                        location=str(raw.get("Location") or raw.get("LOCATION") or ""),
                        terminal1=str(
                            raw.get("terminal-1")
                            or raw.get("TERM1")
                            or raw.get("Term1")
                            or raw.get("Terminal_First_Number")
                            or ""
                        ),
                        terminal2=str(
                            raw.get("terminal-2")
                            or raw.get("TERM2")
                            or raw.get("Term2")
                            or raw.get("Terminal_Second_Number")
                            or ""
                        ),
                        src=str(raw.get("SRC") or raw.get("src") or ""),
                        normalized_tag=str(raw.get("Tag/SPARE") or raw.get("Tag No") or raw.get("normalized_tag") or ""),
                        match_status=str(raw.get("Match") or raw.get("match_status") or raw.get("MATCH") or ""),
                        raw_json=raw,
                    )
                )
            register_artifact(db, run, ArtifactType.FINAL_EXCEL, output_excel_path, mime_type="application/vnd.ms-excel")

        if unmatched_excel_path and os.path.exists(unmatched_excel_path):
            register_artifact(db, run, ArtifactType.UNMATCHED_EXCEL, unmatched_excel_path, mime_type="application/vnd.ms-excel")

        if report_path and os.path.exists(report_path):
            register_artifact(db, run, ArtifactType.REPORT_JSON, report_path, mime_type="application/json")

        if zip_path and os.path.exists(zip_path):
            register_artifact(db, run, ArtifactType.ZIP_BUNDLE, zip_path, mime_type="application/zip")

        for pdf_path in annotated_pdfs:
            if os.path.exists(pdf_path):
                register_artifact(db, run, ArtifactType.ANNOTATED_PDF, pdf_path, mime_type="application/pdf")

        for tag in unmatched_excel_tags:
            db.add(
                Issue(
                    run_id=run.id,
                    project_id=project.id,
                    severity=IssueSeverity.WARNING,
                    status=IssueStatus.OPEN,
                    code="unmatched_excel",
                    message=f"Tag from Excel not matched: {tag}",
                )
            )

        pattern_unmatched_details = pattern_unmatched_details or []
        pattern_unmatched_candidates = pattern_unmatched_candidates or []
        pattern_tags_upper = set()
        for item in pattern_unmatched_details:
            if not isinstance(item, dict):
                continue
            ocr_text = str(item.get("ocr_text") or item.get("display_text") or "").strip()
            if not ocr_text:
                continue
            pattern_tags_upper.add(ocr_text.upper())
            pdf_name = str(item.get("pdf_name") or "").strip()
            page_no = item.get("page")
            page_text = f"{page_no}" if page_no is not None else ""
            page_jb = str(item.get("jb") or "").strip()
            page_mc = str(item.get("mc") or "").strip()
            score = item.get("score")
            reason = str(item.get("reason") or "").strip()
            loc_parts = [p for p in [pdf_name, f"Page {page_text}" if page_text else "", f"JB {page_jb}" if page_jb else ""] if p]
            loc_str = " | ".join(loc_parts)
            message = f"Pattern-like tag not in IO List: {ocr_text}"
            if loc_str:
                message = f"{message} ({loc_str})"
            if score:
                try:
                    message = f"{message} [score={float(score):.2f}]"
                except Exception:
                    pass
            if reason:
                message = f"{message} - {reason}"

            db.add(
                Issue(
                    run_id=run.id,
                    project_id=project.id,
                    severity=IssueSeverity.WARNING,
                    status=IssueStatus.OPEN,
                    code="unmatched_pattern_candidate",
                    message=message,
                    details=to_json_safe(item),
                )
            )

        # fallback: ensure pattern candidates are still represented as addable issues
        # even if detailed page/JB metadata could not be assembled upstream
        for candidate in pattern_unmatched_candidates:
            candidate_text = str(candidate or "").strip()
            candidate_upper = candidate_text.upper()
            if not candidate_text or candidate_upper in pattern_tags_upper:
                continue
            pattern_tags_upper.add(candidate_upper)
            db.add(
                Issue(
                    run_id=run.id,
                    project_id=project.id,
                    severity=IssueSeverity.WARNING,
                    status=IssueStatus.OPEN,
                    code="unmatched_pattern_candidate",
                    message=f"Pattern-like tag not in IO List: {candidate_text}",
                    details=to_json_safe(
                        {
                            "source_type": "pattern_unmatched_candidate",
                            "ocr_text": candidate_text,
                            "display_text": candidate_text,
                            "reason": "Pattern candidate detected in PDF but context metadata was unavailable",
                        }
                    ),
                )
            )

        for tag in unmatched_pdf_tags:
            tag_upper = str(tag or "").strip().upper()
            if tag_upper and tag_upper in pattern_tags_upper:
                continue
            db.add(
                Issue(
                    run_id=run.id,
                    project_id=project.id,
                    severity=IssueSeverity.WARNING,
                    status=IssueStatus.OPEN,
                    code="unmatched_pdf",
                    message=f"Tag from PDF not matched: {tag}",
                )
            )

        run_svc.set_status(db, run, RunStatus.FINALIZED, stage="finalized")
        project.last_finalized_run_id = run.id
        run_svc.add_log_line(db, run, "JBDetection results stored in database", level="info")
        db.flush()
        db.commit()


def process_task_async(task_id, pdf_paths, excel_path, project_name, pattern_config, username, run_id, project_id):
    """پردازش task به صورت asynchronous با مدیریت بهتر وضعیت و heartbeat"""
    
    # Start heartbeat to protect against worker timeout/OOM/SIGKILL
    hb_stop = start_heartbeat(task_id)
    
    try:
        # به‌روزرسانی وضعیت اولیه
        TaskManager.update_task(task_id, {
            'status': TaskStatus.PROCESSING,
            'progress': 10,
            'started_at': datetime.now().isoformat(),
            'last_heartbeat': datetime.now().isoformat(),
            'run_id': run_id,
            'project_id': project_id,
            'worker_pid': os.getpid()  # track which worker is processing
        })

        with session_scope() as db:
            run = db.get(Run, run_id)
            if run:
                run_svc.set_status(db, run, RunStatus.PROCESSING, stage="jbdetection")
                run_svc.add_log_line(db, run, "JBDetection processing started", level="info")
                db.commit()
        
        logger.info(f"Task {task_id}: شروع پردازش برای پروژه {project_name} (worker PID={os.getpid()})")
        
        # ایجاد دایرکتوری‌ها
        project_output_dir = get_project_output_dir(project_name)
        annotated_pdf_dir = os.path.join(project_output_dir, "annotated_pdfs")
        os.makedirs(annotated_pdf_dir, exist_ok=True)
        
        TaskManager.update_task(task_id, {'progress': 20})
        
        # تنظیم فایل‌های خروجی
        output_excel_filename = generate_document_filename(project_name, "Excel", "xlsx")
        output_excel_path = os.path.join(project_output_dir, output_excel_filename)
        
        # ایجاد extractor
        logger.info(f"Task {task_id}: ایجاد extractor")
        extractor = get_platform_specific_extractor(
            tesseract_path=DEFAULT_TESSERACT_PATH,
            excel_path=excel_path
        )

        # Wrap the platform-specific extractor with DataAnalysis for
        # document-type routing and classifier injection.
        data_analysis = DataAnalysis(
            extractor,
            classifier_model_path=PDF_CLASSIFIER_MODEL_PATH,
            classifier_labels_path=PDF_CLASSIFIER_LABELS_PATH,
        )
        extractor = data_analysis
        
        TaskManager.update_task(task_id, {'progress': 30})
        gc.collect()  # free memory before heavy OCR
        
        # تنظیم الگوها
        if hasattr(extractor, 'set_patterns'):
            jb_examples = pattern_config.get('jb_examples', '')
            mc_examples = pattern_config.get('mc_examples', '')
            spare_examples = pattern_config.get('spare_examples', '')
            cable_examples = pattern_config.get('cable_examples', '')
            
            if jb_examples or mc_examples or spare_examples or cable_examples:
                extractor.set_patterns(
                    jb_examples=jb_examples,
                    mc_examples=mc_examples,
                    spare_examples=spare_examples,
                    cable_examples=cable_examples
                )
        
        terminal_pattern = pattern_config.get('terminal_pattern', '')
        wire_color_pattern = pattern_config.get('wire_color_pattern', '')
        
        if terminal_pattern or wire_color_pattern:
            if hasattr(extractor, 'set_terminal_wire_patterns'):
                extractor.set_terminal_wire_patterns(pattern_config)
                logger.info(f"Task {task_id}: الگوهای ترمینال و سیم تنظیم شد")
        
        TaskManager.update_task(task_id, {'progress': 40})
        
        # پردازش فایل‌ها — این بخش طولانی‌ترین قسمت است
        logger.info(f"Task {task_id}: شروع پردازش {len(pdf_paths)} فایل PDF")
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=annotated_pdf_dir
        )
        pattern_unmatched_candidates = list(getattr(extractor, 'latest_pattern_unmatched_candidates', []) or [])
        pattern_unmatched_details = list(getattr(extractor, 'latest_pattern_unmatched_details', []) or [])
        
        TaskManager.update_task(task_id, {'progress': 80})
        logger.info(f"Task {task_id}: پردازش PDF ها کامل شد")
        
        # آزادسازی حافظه بعد از پردازش سنگین OCR
        gc.collect()

        # ایجاد فایل تگ‌های تطبیق نیافته
        unmatched_excel_filename = generate_document_filename(project_name, "UnmatchedTags", "xlsx")
        unmatched_excel_path = os.path.join(project_output_dir, unmatched_excel_filename)
        
        if hasattr(extractor, '_create_unmatched_tags_excel'):
            extractor._create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, unmatched_excel_path)
            logger.info(f"Task {task_id}: فایل تگ‌های تطبیق نیافته ذخیره شد")
        
        # ایجاد گزارش
        report_filename = generate_document_filename(project_name, "Report", "json")
        report_path = os.path.join(project_output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'project_name': project_name,
                'processing_date': datetime.now().isoformat(),
                'user': username,
                'task_id': task_id,
                'patterns': pattern_config,
                'results': {
                    'unmatched_excel_tags': len(unmatched_excel_tags),
                    'unmatched_pdf_tags': len(unmatched_pdf_tags),
                    'pattern_unmatched_candidates': len(pattern_unmatched_candidates),
                    'pattern_unmatched_details': len(pattern_unmatched_details),
                    'pdf_count': len(pdf_paths),
                    'pdf_names': [os.path.basename(p) for p in pdf_paths],
                    'pdf_types': getattr(extractor, 'document_types', {})
                }
            }, f, indent=2, ensure_ascii=False)
        
        # جمع‌آوری فایل‌های خروجی
        output_files = [output_excel_path, unmatched_excel_path, report_path]
        annotated_pdfs = []
        
        for f in os.listdir(annotated_pdf_dir):
            if f.startswith('annotated_'):
                pdf_path = os.path.join(annotated_pdf_dir, f)
                output_files.append(pdf_path)
                annotated_pdfs.append(pdf_path)
        
        TaskManager.update_task(task_id, {'progress': 90})
        
        # ایجاد ZIP
        logger.info(f"Task {task_id}: ایجاد فایل ZIP")
        zip_path = create_zip_archive(project_name, output_files)
        download_url = get_download_url(zip_path)
        
        _persist_run_outputs(
            run_id,
            project_id,
            output_excel_path,
            unmatched_excel_path,
            report_path,
            zip_path,
            annotated_pdfs,
            unmatched_excel_tags,
            unmatched_pdf_tags,
            pattern_unmatched_details,
            pattern_unmatched_candidates,
        )
        
        # ذخیره نتایج نهایی
        final_result = {
            'status': TaskStatus.COMPLETED,
            'progress': 100,
            'completed_at': datetime.now().isoformat(),
            'run_id': run_id,
            'project_id': project_id,
            'result': {
                'output_files': {
                    'excel_path': output_excel_path,
                    'unmatched_excel_path': unmatched_excel_path,
                    'report_path': report_path,
                    'zip_path': zip_path,
                    'download_url': download_url,
                    'annotated_pdfs': annotated_pdfs
                },
                'results': {
                    'unmatched_excel_tags': unmatched_excel_tags,
                    'unmatched_pdf_tags': unmatched_pdf_tags,
                    'pattern_unmatched_candidates': pattern_unmatched_candidates,
                    'pattern_unmatched_details': pattern_unmatched_details,
                    'unmatched_excel_count': len(unmatched_excel_tags),
                    'unmatched_pdf_count': len(unmatched_pdf_tags),
                    'pattern_unmatched_count': len(pattern_unmatched_candidates),
                    'pattern_unmatched_detail_count': len(pattern_unmatched_details)
                },
                'patterns_used': pattern_config
            }
        }
        
        TaskManager.update_task(task_id, final_result)
        logger.info(f"Task {task_id}: پردازش با موفقیت تکمیل شد")
        
    except Exception as e:
        logger.error(f"Task {task_id}: خطا در پردازش - {str(e)}")
        logger.error(traceback.format_exc())

        with session_scope() as db:
            run = db.get(Run, run_id) if run_id else None
            if run:
                run_svc.set_status(db, run, RunStatus.FAILED, stage="jbdetection", notes=str(e))
                run_svc.add_log_line(db, run, f"JBDetection failed: {e}", level="error")
                db.commit()
        
        TaskManager.update_task(task_id, {
            'status': TaskStatus.FAILED,
            'error': str(e),
            'error_details': traceback.format_exc(),
            'progress': 100,
            'failed_at': datetime.now().isoformat()
        })
    finally:
        # Always stop heartbeat when task ends (success or failure)
        stop_heartbeat(task_id)
        # Final memory cleanup
        gc.collect()


def process_io_assignment_task(task_id, excel_path, project_name, config_overrides, username):
    # Start heartbeat to protect against worker timeout/OOM
    hb_stop = start_heartbeat(task_id)
    try:
        io_logger = get_io_assignment_logger(project_name, username)
        TaskManager.update_task(task_id, {
            'status': TaskStatus.PROCESSING,
            'progress': 10,
            'started_at': datetime.now().isoformat(),
            'last_heartbeat': datetime.now().isoformat(),
            'worker_pid': os.getpid()
        })
        append_task_log(task_id, "Task started")
        io_logger.info("Task %s started", task_id)

        project_output_dir = get_project_output_dir(project_name)
        append_task_log(task_id, f"Output directory: {project_output_dir}")
        io_logger.info("Output directory: %s", project_output_dir)
        output_excel_filename = generate_document_filename(project_name, "IOAssignment", "xlsx")
        output_excel_path = os.path.join(project_output_dir, output_excel_filename)

        TaskManager.update_task(task_id, {'progress': 35})
        append_task_log(task_id, "Running IO Assignment engine")
        io_logger.info("Running IO Assignment engine")

        result = run_io_assignment(
            input_excel_path=excel_path,
            output_excel_path=output_excel_path,
            config_overrides=config_overrides
        )

        TaskManager.update_task(task_id, {'progress': 75})
        append_task_log(task_id, "Engine finished, building summaries")
        io_logger.info("Engine finished, building summaries")

        final_df = result['final_df']
        total_active = int((final_df['Signal_Type'] == 'ACTIVE').sum())
        total_hot = int((final_df['Signal_Type'] == 'HOT_SPARE').sum())
        total_spare = int((final_df['Signal_Type'] == 'SPARE').sum())
        total_signals = total_active + total_hot + total_spare
        expected_hot = math.ceil(total_active * result['config'].hot_spare_ratio) if total_active else 0
        hot_compliance = round((total_hot / expected_hot) * 100, 1) if expected_hot else 100.0
        overall_spare_capacity = round(((total_hot + total_spare) / max(total_signals, 1)) * 100, 1)

        summary = {
            'total_active': total_active,
            'total_hot_spares': total_hot,
            'total_spares': total_spare,
            'total_signals': total_signals,
            'expected_hot_spares': expected_hot,
            'hot_spare_compliance': hot_compliance,
            'overall_spare_capacity': overall_spare_capacity,
            'cabinet_count': len(result['cabinets']),
            'jb_count': int(final_df[result['config'].col_mapping['JB']].nunique())
        }

        board_type_col = 'Board_Type'
        board_id_col = 'Board_ID'
        signal_col = 'Signal_Type'
        cabinet_stats = []
        for cab in result['cabinets']:
            cab_df = final_df[final_df['Cabinet_ID'] == cab.id]
            board_counts = cab_df.groupby('Board_Type')['Board_ID'].nunique().to_dict()
            rail_limit = cab.limits.get('Max rail terminals', 0)
            rail_pct = round((cab.rail_used / rail_limit) * 100, 1) if rail_limit else 0.0
            class DummyJB:
                pass
            dummy = DummyJB()
            dummy.channel_counts = {
                'Barrier_AI': 0,
                'Barrier_AO': 0,
                'Barrier_DI': 0,
                'Barrier_DO': 0,
                'Terminal_AI': 0,
                'Terminal_AO': 0,
                'Terminal_DI': 0,
                'Terminal_DO': 0,
                'Relay_DI': 0,
                'Relay_DO': 0,
                'Relay_AI': 0,
                'Relay_AO': 0,
            }
            usage_now, max_total, mode = cab._pool_usage_after(dummy)
            board_slots_used = 0
            board_slots_max = 0
            board_slot_pct = 0.0
            if mode == 'COUNT':
                board_slots_used = int(usage_now or 0)
                board_slots_max = int(max_total or 0)
            elif mode == 'PERCENT_PER_BOARD':
                board_slots_used = round(float(usage_now or 0), 1)
                board_slots_max = round(float(max_total or 0), 1)
            if board_slots_max:
                board_slot_pct = round((float(board_slots_used) / float(board_slots_max)) * 100, 1)

            relay_board_counts = {
                'Relay Board AI capacity': int(math.ceil(cab.channels_relay_ai / result['config'].channels.get('Relay Board AI capacity', 1))) if cab.channels_relay_ai else 0,
                'Relay Board AO capacity': int(math.ceil(cab.channels_relay_ao / result['config'].channels.get('Relay Board AO capacity', 1))) if cab.channels_relay_ao else 0,
                'Relay Board DI capacity': int(math.ceil(cab.channels_relay_di / result['config'].channels.get('Relay Board DI capacity', 1))) if cab.channels_relay_di else 0,
                'Relay Board DO capacity': int(math.ceil(cab.channels_relay_do / result['config'].channels.get('Relay Board DO capacity', 1))) if cab.channels_relay_do else 0,
            }

            board_details = []
            if not cab_df.empty and board_id_col in cab_df.columns:
                for board_id, g in cab_df.groupby(board_id_col):
                    board_type = g.iloc[0][board_type_col] if board_type_col in g.columns else ''
                    match = re.search(r'\(([^)]+)\)', str(board_id))
                    base_io = match.group(1) if match else ''
                    capacity_key = f"{board_type} ({base_io})" if base_io else board_type
                    capacity = result['config'].channels.get(capacity_key, 0)
                    counts = g[signal_col].value_counts().to_dict() if signal_col in g.columns else {}
                    active = int(counts.get('ACTIVE', 0))
                    hot = int(counts.get('HOT_SPARE', 0))
                    spare = int(counts.get('SPARE', 0))
                    total = active + hot + spare
                    fill_pct = round((total / capacity) * 100, 1) if capacity else 0.0
                    board_details.append({
                        'board_id': str(board_id),
                        'board_type': str(board_type),
                        'base_io': base_io,
                        'capacity': int(capacity) if capacity else 0,
                        'total_signals': total,
                        'active_signals': active,
                        'hot_spares': hot,
                        'spare_signals': spare,
                        'fill_pct': fill_pct
                    })
            cabinet_stats.append({
                'cabinet_id': cab.id,
                'cabinet_type': cab.type_name,
                'direction': cab.direction,
                'limiting_factor': cab.limiting_factor,
                'rail_used': int(cab.rail_used),
                'rail_limit': int(rail_limit) if rail_limit else 0,
                'rail_pct': rail_pct,
                'board_slots_used': board_slots_used,
                'board_slots_max': board_slots_max,
                'board_slot_pct': board_slot_pct,
                'board_counts': {k: int(v) for k, v in board_counts.items()},
                'relay_board_counts': relay_board_counts,
                'boards': board_details,
                'jb_list': [jb.name for jb in cab.assigned_jbs][:6],
            })

        report_filename = generate_document_filename(project_name, "IOAssignmentReport", "json")
        report_path = os.path.join(project_output_dir, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'project_name': project_name,
                'processing_date': datetime.now().isoformat(),
                'user': username,
                'task_id': task_id,
                'summary': summary
            }, f, indent=2, ensure_ascii=False)
        append_task_log(task_id, f"Report saved: {report_path}")
        io_logger.info("Report saved: %s", report_path)

        TaskManager.update_task(task_id, {'progress': 90})
        append_task_log(task_id, "Creating ZIP package")
        io_logger.info("Creating ZIP package")

        zip_path = create_zip_archive(project_name, [output_excel_path, report_path], doc_type="IOAssignment")
        download_url = get_download_url(zip_path)
        append_task_log(task_id, f"ZIP created: {zip_path}")
        io_logger.info("ZIP created: %s", zip_path)

        if os.path.exists(excel_path):
            try:
                os.remove(excel_path)
            except Exception as e:
                logger.warning(f"Task {task_id}: خطا در حذف {excel_path}: {e}")

        final_result = {
            'status': TaskStatus.COMPLETED,
            'progress': 100,
            'completed_at': datetime.now().isoformat(),
            'result': {
                'output_files': {
                    'excel_path': output_excel_path,
                    'report_path': report_path,
                    'zip_path': zip_path,
                    'download_url': download_url
                },
                'summary': summary,
                'cabinets': cabinet_stats
            }
        }

        TaskManager.update_task(task_id, to_json_safe(final_result))
        io_logger.info("IO Assignment Task %s completed successfully", task_id)
        append_task_log(task_id, "Task completed")
    except Exception as e:
        logger.error(f"IO Assignment Task {task_id}: خطا در پردازش - {str(e)}")
        logger.error(traceback.format_exc())
        io_logger = get_io_assignment_logger(project_name, username)
        io_logger.error("IO Assignment Task %s failed: %s", task_id, str(e))
        io_logger.error(traceback.format_exc())
        append_task_log(task_id, f"Task failed: {str(e)}")
        TaskManager.update_task(task_id, {
            'status': TaskStatus.FAILED,
            'error': str(e),
            'error_details': traceback.format_exc(),
            'progress': 100,
            'failed_at': datetime.now().isoformat()
        })
    finally:
        # Always stop heartbeat when task ends (success or failure)
        stop_heartbeat(task_id)
        gc.collect()

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('portal'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username in VALID_USERS and VALID_USERS[username] == password:
        session['username'] = username
        global logger
        logger = get_logger('app', username)
        logger.info(f"کاربر {username} وارد سیستم شد")
        return jsonify({'status': 'success'})
    else:
        logger.warning(f"تلاش ناموفق برای ورود با نام کاربری: {username}")
        return jsonify({'status': 'error', 'message': 'نام کاربری یا رمز عبور اشتباه است'})

@app.route('/logout')
def logout():
    username = session.get('username', 'anonymous')
    session.pop('username', None)
    logger.info(f"کاربر {username} از سیستم خارج شد")
    return redirect(url_for('home'))

@app.route('/home')
def portal():
    if 'username' not in session:
        return redirect(url_for('home'))
    username = session.get('username')
    logger.info(f"کاربر {username} به صفحه خانه دسترسی پیدا کرد")
    return render_template('home.html', username=username)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('home'))
    username = session.get('username')
    logger.info(f"کاربر {username} به داشبورد دسترسی پیدا کرد")
    return render_template('JB.html', username=username)

@app.route('/io-assignment')
def io_assignment():
    if 'username' not in session:
        return redirect(url_for('home'))
    username = session.get('username')
    logger.info(f"کاربر {username} به IO Assignment دسترسی پیدا کرد")
    return render_template('io_assignment.html', username=username)

@app.route('/system-info')
def system_info():
    """
    ارائه اطلاعات سیستم و GPU به کاربر
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    
    system_info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'tesseract_path': DEFAULT_TESSERACT_PATH,
        'output_base_dir': BASE_OUTPUT_DIR
    }
    
    try:
        extractor = get_platform_specific_extractor(tesseract_path=DEFAULT_TESSERACT_PATH)
        
        if hasattr(extractor, 'gpu_available'):
            system_info['gpu_available'] = extractor.gpu_available
            if extractor.gpu_available:
                system_info['gpu_type'] = extractor.gpu_type
                if extractor.gpu_type == "NVIDIA" and hasattr(extractor, 'cuda_device_count'):
                    system_info['cuda_device_count'] = extractor.cuda_device_count
    except Exception as e:
        logger.error(f"خطا در دریافت اطلاعات GPU: {e}", extra={'user': username})
        system_info['gpu_error'] = str(e)
    
    logger.info(f"کاربر {username} اطلاعات سیستم را درخواست کرد", extra={'system_info': system_info})
    
    return jsonify({
        'status': 'success',
        'system_info': system_info
    })

@app.route('/task-status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """دریافت وضعیت task — با stale detection خودکار"""
    try:
        if 'username' not in session:
            return jsonify({
                'status': 'error',
                'message': 'لطفاً ابتدا وارد سیستم شوید'
            }), 401

        username = session.get('username')
        task = TaskManager.get_task(task_id)
        
        if not task:
            return jsonify({
                'status': 'error',
                'message': 'Task یافت نشد'
            }), 404

        if task.get('username') != username and not _is_admin_username(username):
            return jsonify({
                'status': 'error',
                'message': 'شما مجاز به مشاهده این task نیستید'
            }), 403
        
        # ── STALE TASK DETECTION — DISABLED IN /task-status ──
        # This was causing FALSE ALARMS: when client polls /task-status during
        # heavy OCR processing, the heartbeat might be slightly delayed (GIL contention),
        # and the task would be incorrectly marked as FAILED.
        #
        # Stale detection now ONLY runs in the background scheduler (every 5 min)
        # with a 30-MINUTE threshold — only truly dead tasks (worker SIGKILL'd) get marked.
        # The /task-status endpoint is now READ-ONLY (does not modify task state).
        # ← stale detection removed — safe read-only status
        
        return jsonify({
            'status': task.get('status', 'pending'),
            'progress': task.get('progress', 0),
            'result': task.get('result'),
            'error': task.get('error'),
            'run_id': task.get('run_id'),
            'project_id': task.get('project_id'),
            'project_name': task.get('project_name', ''),
            'pdf_count': task.get('pdf_count', 0),
            'log': (task.get('log', []) or [])[-20:],  # FIX: Only return last 20 log entries (was returning ALL → 487KB response → socket timeout)
            'started_at': task.get('started_at'),
            'completed_at': task.get('completed_at'),
            'last_heartbeat': task.get('last_heartbeat'),
            'worker_pid': task.get('worker_pid'),
            'failure_reason': task.get('failure_reason')
        })
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/admin/stale-tasks', methods=['POST'])
def cleanup_stale_tasks():
    """Admin endpoint to manually trigger stale task cleanup."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    username = session.get('username')
    if not _is_admin_username(username):
        return jsonify({'status': 'error', 'message': 'Admin access required'}), 403
    marked = mark_stale_tasks_as_failed()
    return jsonify({
        'status': 'success',
        'stale_tasks_marked': marked,
        'message': f'{marked} stale task(s) marked as failed'
    })

@app.route('/admin/task-health', methods=['GET'])
def task_health():
    """Health check endpoint showing active vs stale tasks."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    username = session.get('username')
    if not _is_admin_username(username):
        return jsonify({'status': 'error', 'message': 'Admin access required'}), 403
    
    active = 0
    stale = 0
    completed = 0
    failed = 0
    pending = 0
    try:
        for filename in os.listdir(TASKS_DIR):
            if not filename.endswith('.json') or filename.startswith('.'):
                continue
            task_id = filename[:-5]
            task_data = FileTaskManager.get_task(task_id)
            if not task_data:
                continue
            status = task_data.get('status')
            if status == TaskStatus.COMPLETED:
                completed += 1
            elif status == TaskStatus.FAILED:
                failed += 1
            elif status == TaskStatus.PENDING:
                pending += 1
            elif status == TaskStatus.PROCESSING:
                if is_task_stale(task_data):
                    stale += 1
                else:
                    active += 1
    except Exception as e:
        logger.error(f"Error in task_health: {e}")
    
    return jsonify({
        'status': 'success',
        'active_processing': active,
        'stale_processing': stale,
        'pending': pending,
        'completed': completed,
        'failed': failed,
        'total': active + stale + pending + completed + failed
    })

@app.route('/process', methods=['POST'])
def process_files():
    """شروع پردازش با مدیریت بهتر task"""
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    logger.info(f"کاربر {username} درخواست پردازش فایل‌ها را ارسال کرد")
    
    try:
        # دریافت نام پروژه
        project_name = request.form.get('project_name')
        if not project_name:
            logger.warning(f"کاربر {username} نام پروژه را وارد نکرد")
            return jsonify({
                'status': 'error',
                'message': 'لطفاً نام پروژه را وارد کنید'
            }), 400
        
        # دریافت فایل‌ها
        pdf_files = request.files.getlist('pdf_files')
        excel_file = request.files.get('excel_file')
        
        if not pdf_files or len(pdf_files) == 0:
            return jsonify({
                'status': 'error',
                'message': 'لطفاً حداقل یک فایل PDF انتخاب کنید'
            }), 400
        
        if not excel_file:
            return jsonify({
                'status': 'error',
                'message': 'لطفاً یک فایل Excel انتخاب کنید'
            }), 400
        
        # ذخیره در DB + دیسک و ایجاد Run
        pdf_paths = []
        excel_path = None
        run_id = None
        project_id = None
        with session_scope() as db:
            project = project_svc.get_or_create_project(db, project_name, reuse=True)
            project_id = str(project.id)
            run = run_svc.create_run(db, project, initiated_by=username)
            run_svc.set_status(db, run, RunStatus.PENDING, stage="import")

            stored_pdfs = import_svc.ingest_pdf_files(db, pdf_files, project, run, BASE_OUTPUT_DIR)
            excel_uploaded = import_svc.store_uploaded_file(
                db, excel_file, project, run, UploadedFileType.EXCEL, BASE_OUTPUT_DIR
            )

            pdf_paths = [uf.storage_path for uf in stored_pdfs]
            excel_path = excel_uploaded.storage_path
            run_id = str(run.id)
        
        # دریافت الگوها
        jb_examples = request.form.get('jb_examples', '').strip()
        mc_examples = request.form.get('mc_examples', '').strip()
        spare_examples = request.form.get('spare_examples', '').strip()
        cable_examples = request.form.get('cable_examples', '').strip()
        terminal_pattern = request.form.get('terminal_pattern', '').strip()
        wire_color_pattern = request.form.get('wire_color_pattern', '').strip()
        include_scr = request.form.get('include_scr', 'true').lower() == 'true'
        
        selected_colors_json = request.form.get('selected_colors', '[]')
        try:
            selected_colors = json.loads(selected_colors_json)
        except:
            selected_colors = []
        
        pattern_config = {
            'jb_examples': jb_examples,
            'mc_examples': mc_examples,
            'spare_examples': spare_examples,
            'cable_examples': cable_examples,
            'terminal_pattern': terminal_pattern,
            'wire_color_pattern': wire_color_pattern,
            'include_scr': include_scr,
            'selected_colors': selected_colors
        }
        
        # ایجاد task
        task_id = str(uuid.uuid4())
        
        TaskManager.create_task(task_id, {
            'status': TaskStatus.PENDING,
            'progress': 0,
            'project_name': project_name,
            'project_id': project_id,
            'run_id': run_id,
            'username': username,
            'pdf_count': len(pdf_paths),
            'pdf_names': [os.path.basename(p) for p in pdf_paths]
        })
    
        
        logger.info(f"Task {task_id} ایجاد شد برای کاربر {username} - پروژه: {project_name}")
        
        # شروع پردازش در thread جداگانه
        thread = threading.Thread(
            target=process_task_async,
            args=(task_id, pdf_paths, excel_path, project_name, pattern_config, username, run_id, project_id),
            daemon=False  # تغییر به False برای اطمینان از تکمیل پردازش
        )
        thread.start()
        
        # برگرداندن task_id به کلاینت
        return jsonify({
            'status': 'success',
            'message': 'پردازش آغاز شد',
            'task_id': task_id,
            'project_name': project_name,
            'run_id': run_id,
            'project_id': project_id
        })
        
    except Exception as e:
        logger.error(f"خطا در شروع پردازش: {str(e)}", extra={'user': username})
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': {
                'error_type': type(e).__name__,
                'error_description': str(e)
            }
        }), 500

@app.route('/api/process', methods=['POST'])
def api_process():
    """
    API endpoint برای پردازش فایل‌ها
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    logger.info(f"کاربر {username} درخواست API پردازش فایل‌ها را ارسال کرد")
    
    try:
        data = request.json
        pdf_paths = data.get('pdf_paths', [])
        excel_path = data.get('excel_path')
        project_name = data.get('project_name')
        
        if not pdf_paths or not excel_path:
            return jsonify({
                'status': 'error',
                'message': 'مسیرهای PDF و Excel الزامی هستند'
            }), 400
        
        if not project_name:
            return jsonify({
                'status': 'error',
                'message': 'نام پروژه الزامی است'
            }), 400
        
        project_output_dir = get_project_output_dir(project_name)
        log_dir = get_log_dir(project_name)
        
        output_excel_filename = generate_document_filename(project_name, "Excel", "xlsx")
        output_excel_path = os.path.join(project_output_dir, output_excel_filename)
        
        annotated_pdf_dir = os.path.join(project_output_dir, "annotated_pdfs")
        os.makedirs(annotated_pdf_dir, exist_ok=True)
        
        extractor = get_platform_specific_extractor(
            tesseract_path=DEFAULT_TESSERACT_PATH,
            excel_path=excel_path
        )
        
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=annotated_pdf_dir
        )
        pattern_unmatched_candidates = list(getattr(extractor, 'latest_pattern_unmatched_candidates', []) or [])
        pattern_unmatched_details = list(getattr(extractor, 'latest_pattern_unmatched_details', []) or [])
        
        unmatched_excel_filename = generate_document_filename(project_name, "UnmatchedTags", "xlsx")
        unmatched_excel_path = os.path.join(project_output_dir, unmatched_excel_filename)
        
        if hasattr(extractor, '_create_unmatched_tags_excel'):
            extractor._create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, unmatched_excel_path)
        
        output_files = [output_excel_path, unmatched_excel_path]
        
        annotated_pdfs = []
        for f in os.listdir(annotated_pdf_dir):
            if f.startswith('annotated_'):
                pdf_path = os.path.join(annotated_pdf_dir, f)
                output_files.append(pdf_path)
                annotated_pdfs.append(pdf_path)
        
        zip_path = create_zip_archive(project_name, output_files)
        download_url = get_download_url(zip_path)
        
        response = {
            "status": "success",
            "message": "Processing completed successfully",
            "details": {
                "output_files": {
                    "excel_path": output_excel_path,
                    "annotated_pdfs": annotated_pdfs,
                    "zip_path": zip_path,
                    "download_url": download_url
                },
                "results": {
                    "unmatched_pdf_tags": unmatched_pdf_tags,
                    "unmatched_excel_tags": unmatched_excel_tags,
                    "pattern_unmatched_candidates": pattern_unmatched_candidates,
                    "pattern_unmatched_details": pattern_unmatched_details
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"خطا در API پردازش فایل‌ها: {str(e)}", extra={'user': username})
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/io-assignment/process', methods=['POST'])
def process_io_assignment():
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401

    username = session.get('username')

    try:
        project_name = request.form.get('project_name', '').strip()
        project_id = request.form.get('project_id', '').strip()
        jb_project_name = request.form.get('jb_project_name', '').strip()
        excel_file = request.files.get('excel_file')
        use_jb_output = request.form.get('use_jb_output', 'false').lower() == 'true'
        config_json = request.form.get('config', '')

        if not project_name:
            return jsonify({'status': 'error', 'message': 'نام پروژه الزامی است'}), 400

        # اگر قرار است از خروجی JB استفاده شود
        temp_path = None
        if use_jb_output:
            if not jb_project_name and not project_id:
                return jsonify({'status': 'error', 'message': 'نام پروژه JB برای استفاده از خروجی DB الزامی است'}), 400

            # Preferred path: lookup by project name (exact first, then partial; latest finalized run wins).
            temp_path = get_latest_excel_from_db(project_name=jb_project_name, username=username) if jb_project_name else None
            if not temp_path and project_id:
                # backward compatibility for older clients
                temp_path = get_latest_excel_from_db(project_id=project_id, username=username)

            if not temp_path:
                return jsonify({'status': 'error', 'message': 'خروجی Excel نهایی برای این پروژه یافت نشد'}), 404
        else:
            if not excel_file or excel_file.filename == '':
                return jsonify({'status': 'error', 'message': 'فایل Excel الزامی است'}), 400
            filename = secure_filename(excel_file.filename)
            unique_name = f"io_{uuid.uuid4().hex}_{filename}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            excel_file.save(temp_path)
            logger.info(f"IO Assignment upload saved: {temp_path}", extra={'user': username})

        config_overrides = {}
        if config_json:
            try:
                config_overrides = json.loads(config_json)
            except json.JSONDecodeError:
                return jsonify({'status': 'error', 'message': 'فرمت تنظیمات نامعتبر است'}), 400

        task_id = str(uuid.uuid4())
        TaskManager.create_task(task_id, {
            'status': TaskStatus.PENDING,
            'progress': 0,
            'task_id': task_id,
            'project_name': project_name,
            'username': username,
            'task_type': 'io_assignment',
            'created_at': datetime.now().isoformat()
        })
        append_task_log(task_id, "Task queued")

        thread = threading.Thread(
            target=process_io_assignment_task,
            args=(task_id, temp_path, project_name, config_overrides, username),
            daemon=True
        )
        thread.start()

        return jsonify({'status': 'success', 'task_id': task_id})
    except Exception as e:
        logger.error(f"خطا در شروع IO Assignment: {e}", extra={'user': username})
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500

OUTPUT_DIRS = {
    "v1": "/home/devio/JB-outputs",
    "v2": "/home/devio/JB-outputs"
}


# ═══════════════════════════════════════════════════════════════════════════
# SIEMENS MODE — lightweight tag-only matching for digital PDFs
# ═══════════════════════════════════════════════════════════════════════════
# This mode is intentionally separate from the Honeywell (JB/MC) pipeline.
# It only takes an IO List + digital PDFs, finds tag matches in the PDF's
# digital text layer, and draws bounding boxes. No OCR, no classifier, no
# JB/MC patterns, no page filtering. See siemens_mode.py for full docs.
try:
    from siemens_mode import SiemensModeProcessor
    _siemens_processor = None  # lazy singleton
    def _get_siemens_processor():
        global _siemens_processor
        if _siemens_processor is None:
            _siemens_processor = SiemensModeProcessor()
        return _siemens_processor
    logger.info("Siemens mode module loaded successfully")
except Exception as _siemens_import_err:
    logger.warning("Siemens mode module not available: %s", _siemens_import_err)
    _get_siemens_processor = None


@app.route('/api/process/siemens', methods=['POST'])
def process_siemens_mode():
    """
    POST /api/process/siemens

    Accepts multipart form data:
      - pdf_files: one or more PDF files (digital PDFs only)
      - excel_file: the IO List Excel file
      - project_name: project name (used for organizing output directory)

    Returns JSON:
      {
        "status": "success",
        "result": { ... SiemensResult.to_dict() ... }
      }
    or:
      { "status": "error", "message": "..." }
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'لطفاً ابتدا وارد سیستم شوید'}), 401

    if _get_siemens_processor is None:
        return jsonify({
            'status': 'error',
            'message': 'سیستم Siemens mode در دسترس نیست — ماژول siemens_mode بارگذاری نشده'
        }), 503

    username = session.get('username')

    try:
        project_name = request.form.get('project_name', '').strip()
        if not project_name:
            return jsonify({'status': 'error', 'message': 'نام پروژه الزامی است'}), 400

        pdf_files = request.files.getlist('pdf_files')
        if not pdf_files or all(f.filename == '' for f in pdf_files):
            return jsonify({'status': 'error', 'message': 'حداقل یک فایل PDF الزامی است'}), 400

        excel_file = request.files.get('excel_file')
        if not excel_file or excel_file.filename == '':
            return jsonify({'status': 'error', 'message': 'فایل Excel الزامی است'}), 400

        # Save uploaded files to temp locations
        safe_project = re.sub(r'[^A-Za-z0-9_-]+', '_', project_name)[:50]
        run_id = uuid.uuid4().hex[:8]
        run_dir = os.path.join(BASE_OUTPUT_DIR, 'siemens_runs', f"{safe_project}_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        pdf_paths = []
        for pdf in pdf_files:
            fn = secure_filename(pdf.filename)
            save_path = os.path.join(run_dir, fn)
            pdf.save(save_path)
            pdf_paths.append(save_path)

        excel_filename = secure_filename(excel_file.filename)
        excel_path = os.path.join(run_dir, excel_filename)
        excel_file.save(excel_path)

        output_pdf_dir = os.path.join(run_dir, 'annotated_pdfs')
        output_excel_path = os.path.join(run_dir, f"siemens_match_report_{run_id}.xlsx")

        logger.info(
            "Siemens mode started: user=%s project=%s pdfs=%d excel=%s",
            username, project_name, len(pdf_paths), excel_filename,
            extra={'user': username}
        )

        # Run synchronously — for production you may want to push this to a
        # background worker like the Honeywell mode does, but for digital
        # PDFs (no OCR) it's typically very fast (1-5 seconds per 100 pages).
        proc = _get_siemens_processor()
        result = proc.process(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_pdf_dir=output_pdf_dir,
            output_excel_path=output_excel_path,
            create_zip=True,
            zip_path=os.path.join(run_dir, f"siemens_bundle_{run_id}.zip"),
        )

        result_dict = result.to_dict()
        result_dict['run_dir'] = run_dir
        result_dict['project_name'] = project_name
        result_dict['username'] = username

        logger.info(
            "Siemens mode finished: project=%s matched=%d/%d (%.1f%%) duration=%.2fs",
            project_name,
            len(result.matched_tags),
            result.total_io_tags,
            (len(result.matched_tags) / result.total_io_tags * 100) if result.total_io_tags else 0,
            result.duration_seconds,
            extra={'user': username}
        )

        return jsonify({
            'status': 'success',
            'result': result_dict,
        })

    except Exception as exc:
        logger.error("Siemens mode failed: %s", exc, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'خطا در پردازش: {exc}'
        }), 500


@app.route('/api/siemens/status', methods=['GET'])
def siemens_mode_status():
    """
    GET /api/siemens/status — quick health check for the Siemens mode module.
    Returns whether the module is loaded and ready to accept requests.
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    return jsonify({
        'status': 'success',
        'available': _get_siemens_processor is not None,
        'mode': 'siemens',
        'description': 'Tag-only matching for digital PDFs — reuses DataAnalysisModule, no JB/MC patterns'
    })


@app.route('/api/siemens/download/<path:filename>', methods=['GET'])
def siemens_download(filename):
    """
    Serve files from the Siemens mode output directory.
    Unlike /download, this does NOT require the file to be registered in the
    ExportArtifact DB table — it only checks the session.

    The filename parameter is the path relative to BASE_OUTPUT_DIR/siemens_runs/.
    For example: /api/siemens/download/myproject_abc123/annotated_pdfs/annotated_test.pdf
    """
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    username = session.get('username')

    try:
        # Security: resolve the path and verify it's under siemens_runs
        siemens_base = os.path.join(BASE_OUTPUT_DIR, 'siemens_runs')
        full_path = os.path.normpath(os.path.join(siemens_base, filename))
        # Prevent directory traversal
        if not full_path.startswith(siemens_base):
            return jsonify({'error': 'Access denied'}), 403

        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'error': f'File not found: {filename}'}), 404

        directory = os.path.dirname(full_path)
        basename = os.path.basename(full_path)
        logger.info(f"Siemens download: user={username} file={basename}")
        return send_from_directory(directory, basename, as_attachment=True)
    except Exception as e:
        logger.error(f"Siemens download error: {e}", extra={'user': username})
        return jsonify({'error': str(e)}), 500


@app.route('/download', methods=['GET'])
def download_file():
    """
    دانلود فایل با مسیر نسبی مشخص شده
    
    پارامترها:
        file: مسیر نسبی فایل برای دانلود (نسبت به دایرکتوری خروجی)
        
    Returns:
        فایل برای دانلود
    """
    try:
        if 'username' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        username = session.get('username')
        file_path = request.args.get('file')
        
        if not file_path:
            return jsonify({"error": "No file path provided"}), 400
        
        # حذف اسلش اضافی از ابتدای مسیر
        if file_path.startswith('/'):
            file_path = file_path[1:]
        
        # تعیین نسخه سرویس (v1 یا v2) بر اساس پورت درخواست
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
        
        # مسیر کامل فایل در سیستم میزبان
        base_dir = OUTPUT_DIRS[version]
        full_path = os.path.join(base_dir, file_path)
        
        # بررسی امنیتی: اطمینان از اینکه فایل درخواستی در مسیر مجاز قرار دارد
        abs_path = os.path.abspath(full_path)
        if not abs_path.startswith(base_dir):
            return jsonify({"error": "Access denied"}), 403
        
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return jsonify({"error": f"File not found: {abs_path}"}), 404

        if not _is_admin_username(username):
            with session_scope() as db:
                permitted = db.scalar(
                    select(ExportArtifact.id)
                    .join(Run, Run.id == ExportArtifact.run_id)
                    .where(ExportArtifact.storage_path == abs_path, Run.initiated_by == username)
                    .limit(1)
                )
                if not permitted:
                    task_paths = _collect_user_task_file_paths(username)
                    if abs_path not in task_paths:
                        return jsonify({"error": "Access denied"}), 403
        
        # تعیین نام فایل برای دانلود
        filename = os.path.basename(abs_path)
        directory = os.path.dirname(abs_path)
        
        logger.info(f"کاربر {username} درخواست دانلود فایل {filename} را ارسال کرد")
        
        # ارسال فایل برای دانلود
        return send_from_directory(directory, filename, as_attachment=True)
        
    except Exception as e:
        username = session.get('username', 'anonymous')
        logger.error(f"خطا در دانلود فایل: {str(e)}", extra={'user': username})
        return jsonify({"error": str(e)}), 500


@app.route('/downloads/<path:relpath>', methods=['GET'])
def download_artifact(relpath):
    """
    سرو فایل‌های خروجی ذخیره شده در BASE_OUTPUT_DIR از مسیر /downloads/...
    """
    try:
        if 'username' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        username = session.get('username')
        # مشابه منطق /download ولی با الگوی مسیری که get_download_url تولید می‌کند
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
        base_dir = OUTPUT_DIRS.get(version, BASE_OUTPUT_DIR)
        full_path = os.path.join(base_dir, relpath)
        abs_path = os.path.abspath(full_path)
        if not abs_path.startswith(base_dir):
            return jsonify({"error": "Access denied"}), 403
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return jsonify({"error": f"File not found: {abs_path}"}), 404

        if not _is_admin_username(username):
            with session_scope() as db:
                permitted = db.scalar(
                    select(ExportArtifact.id)
                    .join(Run, Run.id == ExportArtifact.run_id)
                    .where(ExportArtifact.storage_path == abs_path, Run.initiated_by == username)
                    .limit(1)
                )
                if not permitted:
                    task_paths = _collect_user_task_file_paths(username)
                    if abs_path not in task_paths:
                        return jsonify({"error": "Access denied"}), 403
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        logger.info(f"کاربر {username} دانلود فایل {filename} را از /downloads درخواست کرد")
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        username = session.get('username', 'anonymous')
        logger.error(f"خطا در /downloads: {str(e)}", extra={'user': username})
        return jsonify({"error": str(e)}), 500

@app.route('/download-all-pdfs', methods=['GET'])
def download_all_pdfs():
    """
    دانلود همه فایل‌های PDF یک پروژه به صورت فشرده
    
    پارامترها:
        project: نام پروژه
        
    Returns:
        فایل ZIP حاوی همه PDF‌ها
    """
    try:
        if 'username' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        username = session.get('username')
        project_name = request.args.get('project')
        
        if not project_name:
            return jsonify({"error": "No project name provided"}), 400
        
        # تعیین نسخه سرویس (v1 یا v2) بر اساس پورت درخواست
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
        
        base_dir = OUTPUT_DIRS[version]
        with session_scope() as db:
            run_stmt = (
                select(Run.id)
                .join(Project, Project.id == Run.project_id)
                .where(Project.project_name == project_name)
            )
            if not _is_admin_username(username):
                run_stmt = run_stmt.where(Run.initiated_by == username)
            run_ids = [rid for rid in db.scalars(run_stmt).all()]

            if not run_ids:
                return jsonify({"error": "Project not found or access denied"}), 404

            pdf_files = [
                p for p in db.scalars(
                    select(ExportArtifact.storage_path)
                    .where(
                        ExportArtifact.run_id.in_(run_ids),
                        ExportArtifact.artifact_type == ArtifactType.ANNOTATED_PDF
                    )
                ).all()
                if p and os.path.exists(p) and os.path.isfile(p)
            ]
        
        if not pdf_files:
            return jsonify({"error": "No PDF files found for this project"}), 404
        
        # ایجاد فایل ZIP موقت
        safe_project = secure_filename(project_name) or "project"
        safe_user = secure_filename(username) or "user"
        zip_filename = f"{safe_project}_{safe_user}_PDFs.zip"
        zip_path = os.path.join(base_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pdf_file in pdf_files:
                # افزودن فایل با نام نسبی (بدون مسیر کامل)
                arcname = os.path.basename(pdf_file)
                zipf.write(pdf_file, arcname)
        
        logger.info(f"کاربر {username} درخواست دانلود همه PDF های پروژه {project_name} را ارسال کرد")
        
        # ارسال فایل ZIP برای دانلود
        return send_from_directory(base_dir, zip_filename, as_attachment=True)
        
    except Exception as e:
        username = session.get('username', 'anonymous')
        logger.error(f"خطا در دانلود همه PDF ها: {str(e)}", extra={'user': username})
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """
    لیست همه task‌های کاربر فعلی
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
        
    username = session.get('username')
    tasks = TaskManager.list_user_tasks(username)
    
    return jsonify({
        'status': 'success',
        'tasks': tasks
    })

@app.route('/api/task/<task_id>/delete', methods=['DELETE'])
def delete_task(task_id):
    """
    حذف یک task از لیست
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
        
    username = session.get('username')
    task = TaskManager.get_task(task_id)
    
    if not task:
        return jsonify({
            'status': 'error',
            'message': 'Task یافت نشد'
        }), 404
        
    if task.get('username') != username:
        return jsonify({
            'status': 'error',
            'message': 'شما مجاز به حذف این task نیستید'
        }), 403
        
    success = TaskManager.delete_task(task_id)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Task با موفقیت حذف شد'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'خطا در حذف task'
        }), 500

# cleanup function
def cleanup_old_tasks():
    """حذف task‌های قدیمی + تشخیص task‌های stale."""
    TaskManager.cleanup_old_tasks()
    # Also detect and mark stale tasks (worker died from timeout/OOM)
    try:
        mark_stale_tasks_as_failed()
    except Exception as e:
        logger.error(f"Error in stale task cleanup: {e}")

# اجرای cleanup هر ساعت
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
# Old cleanup: every 2 hours
scheduler.add_job(func=cleanup_old_tasks, trigger="interval", hours=2)
# Stale task detection: every 60 seconds (catches worker deaths quickly)
scheduler.add_job(
    func=mark_stale_tasks_as_failed,
    trigger="interval",
    seconds=HEARTBEAT_CHECK_INTERVAL_SECONDS,
    id='stale_task_detector',
    name='Detect stale tasks (worker timeout/OOM)'
)
scheduler.start()

# خاموش کردن scheduler هنگام خروج
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    # Print startup message
    print("=" * 50)
    print("JB Detection System")
    print("=" * 50)
    print(f"سیستم عامل: {platform.system()}")
    print(f"مسیر Tesseract: {DEFAULT_TESSERACT_PATH}")
    
    # ایجاد پوشه پشتیبان‌گیری اگر وجود ندارد
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"مسیر خروجی: {BASE_OUTPUT_DIR}")
    
    # بررسی وضعیت GPU
    try:
        extractor = get_platform_specific_extractor(tesseract_path=DEFAULT_TESSERACT_PATH)
        if hasattr(extractor, 'gpu_available') and extractor.gpu_available:
            print(f"GPU در دسترس: {extractor.gpu_type}")
            if extractor.gpu_type == "NVIDIA":
                print(f"تعداد دستگاه‌های CUDA: {extractor.cuda_device_count}")
        else:
            print("GPU در دسترس نیست، از پردازش CPU استفاده می‌شود")
    except Exception as e:
        print(f"خطا در بررسی وضعیت GPU: {e}")
    
    logger.info("سرور راه‌اندازی شد")
    print("در حال راه‌اندازی سرور...")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
