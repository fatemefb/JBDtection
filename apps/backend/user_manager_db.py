"""
user_manager_db.py — PostgreSQL-backed multi-user management for Control IntelliCraft.

Replaces user_manager.py (JSON-based) with a fully DB-backed implementation.
Tables (defined in db/models.py):
  • users                — user accounts (bcrypt password hashes)
  • user_sessions        — server-side session tracking
  • audit_logs           — append-only audit trail
  • io_assignment_presets — per-user cabinet configuration presets

Features:
  • bcrypt password hashing (rounds=12)
  • Server-side session tokens (stored in DB, not just Flask cookie)
  • Rate limiting per user (failed_login_count + locked_until)
  • Force-logout (admin can revoke any active session)
  • Audit log for every security-relevant action
  • Three roles: admin > engineer > viewer
  • Backward-compatible: _is_admin_username() still works

Dependencies:
  • bcrypt (pip install bcrypt)
  • apps.backend.db.session (SessionLocal, session_scope)
  • apps.backend.db.models (User, UserSession, AuditLog, IOAssignmentPreset, UserRole, ...)

Usage in app.py:
    from user_manager_db import (
        authenticate, get_current_user, require_role,
        list_users, create_user, update_user, delete_user,
        list_active_sessions, revoke_session,
        log_audit,
    )

    @app.route('/login', methods=['POST'])
    def login():
        user = authenticate(request.form.get('username'), request.form.get('password'))
        if user:
            session['user_id'] = user['id']
            session['session_token'] = user['session_token']  # server-side session
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'نام کاربری یا رمز اشتباه'})
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from flask import jsonify, redirect, request, session, url_for

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────
SESSION_DURATION_HOURS = int(os.environ.get("SESSION_DURATION_HOURS", "12"))
FAILED_LOGIN_LIMIT = 5
LOCKOUT_DURATION_MINUTES = 30

# Role hierarchy: admin > engineer > viewer
_ROLE_LEVEL = {"viewer": 1, "engineer": 2, "admin": 3}


def _role_level(role) -> int:
    """Get numeric level for a role. Accepts string or UserRole enum."""
    if hasattr(role, "value"):
        role = role.value
    return _ROLE_LEVEL.get(str(role), 0)


# ── Lazy imports (so this module loads even if DB isn't available yet) ─────
_SessionLocal = None
_Models = None


def _get_db():
    """Lazy access to SessionLocal."""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    try:
        from apps.backend.db.session import SessionLocal
        _SessionLocal = SessionLocal
        return _SessionLocal
    except Exception:
        try:
            from db.session import SessionLocal  # type: ignore
            _SessionLocal = SessionLocal
            return _SessionLocal
        except Exception as exc:
            logger.error("Cannot import SessionLocal: %s", exc)
            raise


def _get_models():
    """Lazy access to models module."""
    global _Models
    if _Models is not None:
        return _Models
    try:
        from apps.backend.db import models
        _Models = models
        return _Models
    except Exception:
        try:
            from db import models  # type: ignore
            _Models = models
            return _Models
        except Exception as exc:
            logger.error("Cannot import models: %s", exc)
            raise


# ── Audit logging ──────────────────────────────────────────────────────────
def log_audit(
    username: Optional[str],
    action: str,
    target_username: str = None,
    target_resource: str = None,
    success: bool = True,
    message: str = None,
    details: Dict = None,
    ip_address: str = None,
    user_agent: str = None,
    user_id: UUID = None,
) -> None:
    """Write an entry to the audit_logs table. Best-effort — never raises."""
    try:
        models = _get_models()
        SessionLocal = _get_db()
        db = SessionLocal()
        try:
            # Resolve action enum value
            try:
                action_enum = models.AuditAction(action)
            except ValueError:
                action_enum = models.AuditAction.PERMISSION_DENIED

            entry = models.AuditLog(
                user_id=user_id,
                username=username,
                action=action_enum,
                target_username=target_username,
                target_resource=target_resource,
                ip_address=ip_address or (request.remote_addr if request else None),
                user_agent=user_agent or (request.headers.get("User-Agent")[:500] if request else None),
                success=success,
                message=message,
                details=details,
            )
            db.add(entry)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.error("Failed to write audit log: %s", exc)


# ── Authentication ─────────────────────────────────────────────────────────
def _is_locked(user) -> Tuple[bool, str]:
    """Check if a user account is locked due to failed logins."""
    if not user.locked_until:
        return False, ""
    if datetime.utcnow() < user.locked_until.replace(tzinfo=None):
        remaining = int((user.locked_until.replace(tzinfo=None) - datetime.utcnow()).total_seconds() / 60)
        return True, f"حساب قفل است. {remaining} دقیقه دیگر تلاش کنید."
    # Lock expired — reset
    return False, ""


def _record_failed_login(user) -> None:
    """Increment failed_login_count; lock if limit exceeded."""
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        user.failed_login_count = (user.failed_login_count or 0) + 1
        if user.failed_login_count >= FAILED_LOGIN_LIMIT:
            user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            logger.warning("User '%s' locked after %d failed attempts until %s",
                          user.username, user.failed_login_count, user.locked_until)
        db.merge(user)
        db.commit()
    finally:
        db.close()


def _record_successful_login(user, ip: str) -> None:
    """Reset failed_login_count, update last_login_at/ip."""
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        user.failed_login_count = 0
        user.locked_until = None
        user.last_login_at = datetime.utcnow()
        user.last_login_ip = ip
        db.merge(user)
        db.commit()
    finally:
        db.close()


def _create_session(user, ip: str, user_agent: str) -> str:
    """Create a new UserSession row and return the session_token."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        token = secrets.token_urlsafe(48)
        user_session = models.UserSession(
            user_id=user.id,
            session_token=token,
            status=models.SessionStatus.ACTIVE,
            ip_address=ip,
            user_agent=user_agent[:500] if user_agent else None,
            expires_at=datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS),
        )
        db.add(user_session)
        db.commit()
        return token
    finally:
        db.close()


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user. On success, returns user dict + session_token.
    On failure, returns None. Records all attempts to audit log.
    """
    if not username or not password:
        return None

    models = _get_models()
    SessionLocal = _get_db()
    ip = request.remote_addr if request else "unknown"
    user_agent = request.headers.get("User-Agent", "") if request else ""

    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.username == username).first()
    finally:
        db.close()

    if not user:
        log_audit(username=username, action="login_failed",
                   message=f"Unknown username: {username}", ip_address=ip,
                   success=False)
        return None

    if not user.is_active:
        log_audit(username=username, action="login_failed", user_id=user.id,
                   message="Account inactive", ip_address=ip, success=False)
        return None

    locked, reason = _is_locked(user)
    if locked:
        log_audit(username=username, action="login_failed", user_id=user.id,
                   message=reason, ip_address=ip, success=False)
        return None

    if not user.check_password(password):
        _record_failed_login(user)
        log_audit(username=username, action="login_failed", user_id=user.id,
                   message="Wrong password", ip_address=ip, success=False)
        return None

    # Success
    _record_successful_login(user, ip)
    session_token = _create_session(user, ip, user_agent)
    log_audit(username=username, action="login", user_id=user.id,
               message="Login successful", ip_address=ip, success=True)

    return {
        "id": str(user.id),
        "username": user.username,
        "role": user.role.value if hasattr(user.role, "value") else str(user.role),
        "display_name": user.display_name or user.username,
        "email": user.email or "",
        "session_token": session_token,
    }


# ── Session helpers ────────────────────────────────────────────────────────
def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Get the currently logged-in user. Validates the server-side session token
    against the DB (so revoked/expired sessions are caught).
    Returns None if not logged in or session invalid.
    """
    user_id = session.get("user_id")
    session_token = session.get("session_token")
    if not user_id or not session_token:
        return None

    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        # Validate session token
        user_session = db.query(models.UserSession).filter(
            models.UserSession.session_token == session_token,
            models.UserSession.user_id == user_id,
        ).first()

        if not user_session or user_session.status != models.SessionStatus.ACTIVE:
            # Session revoked or logged out — clear Flask session
            session.clear()
            return None

        # Check expiry
        if user_session.expires_at and datetime.utcnow() > user_session.expires_at.replace(tzinfo=None):
            user_session.status = models.SessionStatus.EXPIRED
            db.commit()
            session.clear()
            return None

        # Update last_activity_at
        user_session.last_activity_at = datetime.utcnow()
        db.commit()

        # Get user (re-check active status)
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user or not user.is_active:
            session.clear()
            return None

        return {
            "id": str(user.id),
            "username": user.username,
            "role": user.role.value if hasattr(user.role, "value") else str(user.role),
            "display_name": user.display_name or user.username,
            "email": user.email or "",
            "session_token": session_token,
        }
    finally:
        db.close()


def logout_current_user() -> None:
    """Mark the current session as logged_out and clear Flask session."""
    session_token = session.get("session_token")
    username = session.get("username")

    if session_token:
        models = _get_models()
        SessionLocal = _get_db()
        db = SessionLocal()
        try:
            user_session = db.query(models.UserSession).filter(
                models.UserSession.session_token == session_token
            ).first()
            if user_session:
                user_session.status = models.SessionStatus.LOGGED_OUT
                user_session.logged_out_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    if username:
        log_audit(username=username, action="logout",
                   ip_address=request.remote_addr if request else None)

    session.clear()


def is_admin() -> bool:
    """Check if current user is admin (shortcut)."""
    user = get_current_user()
    return user is not None and user["role"] == "admin"


def is_admin_username(username: Optional[str]) -> bool:
    """Backward-compat: check if a username has admin role."""
    if not username:
        return False
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.username == str(username).strip()).first()
        if not user:
            return False
        role = user.role.value if hasattr(user.role, "value") else str(user.role)
        return role == "admin"
    finally:
        db.close()


def is_logged_in() -> bool:
    """Check if any user is logged in."""
    return get_current_user() is not None


# ── Role-based access control ──────────────────────────────────────────────
def require_role(min_role: str) -> Callable:
    """Decorator: require the current user to have at least `min_role`."""
    min_level = _role_level(min_role)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                if request.method == "GET" and not request.is_json:
                    return redirect(url_for("home"))
                return jsonify({
                    "status": "error",
                    "message": "لطفاً ابتدا وارد سیستم شوید"
                }), 401

            if _role_level(user["role"]) < min_level:
                log_audit(
                    username=user["username"],
                    action="permission_denied",
                    message=f"Access denied to {request.path} (role={user['role']}, required={min_role})",
                    ip_address=request.remote_addr if request else None,
                    success=False,
                )
                return jsonify({
                    "status": "error",
                    "message": f"دسترسی denied. نقش مورد نیاز: {min_role}"
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator


def require_login() -> Callable:
    """Decorator: require the user to be logged in (any role)."""
    return require_role("viewer")


# ── User CRUD ──────────────────────────────────────────────────────────────
def list_users() -> List[Dict[str, Any]]:
    """Return all users (without password hashes)."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        users = db.query(models.User).order_by(models.User.created_at.desc()).all()
        return [u.to_dict() for u in users]
    finally:
        db.close()


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get a single user by username."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.username == username).first()
        return user.to_dict() if user else None
    finally:
        db.close()


def create_user(username: str, password: str, role: str = "engineer",
                display_name: str = None, email: str = None,
                created_by: str = None) -> Tuple[bool, str]:
    """Create a new user. Returns (success, message)."""
    username = (username or "").strip()
    if not username or not password:
        return False, "نام کاربری و رمز عبور الزامی است"
    if len(password) < 4:
        return False, "رمز عبور باید حداقل ۴ کاراکتر باشد"
    if role not in ("admin", "engineer", "viewer"):
        return False, f"نقش نامعتبر: {role}"

    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        existing = db.query(models.User).filter(models.User.username == username).first()
        if existing:
            return False, f"کاربر '{username}' قبلاً وجود دارد"

        try:
            role_enum = models.UserRole(role)
        except ValueError:
            role_enum = models.UserRole.ENGINEER

        user = models.User(
            username=username,
            role=role_enum,
            display_name=display_name or username,
            email=email or "",
            is_active=True,
            created_by=created_by,
        )
        user.set_password(password)
        db.add(user)
        db.commit()

        log_audit(username=created_by, action="user_create",
                   target_username=username, target_resource=str(user.id),
                   message=f"User '{username}' created with role '{role}'",
                   ip_address=request.remote_addr if request else None)
        return True, f"کاربر '{username}' با موفقیت ایجاد شد"
    except Exception as exc:
        db.rollback()
        logger.error("Failed to create user: %s", exc)
        return False, f"خطا: {exc}"
    finally:
        db.close()


def update_user(username: str, password: str = None, role: str = None,
                display_name: str = None, email: str = None,
                active: bool = None, updated_by: str = None) -> Tuple[bool, str]:
    """Update an existing user. Only updates fields that are not None."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.username == username).first()
        if not user:
            return False, f"کاربر '{username}' یافت نشد"

        changes = []
        if password and password.strip():
            user.set_password(password)
            changes.append("password")
        if role and role in ("admin", "engineer", "viewer"):
            try:
                user.role = models.UserRole(role)
                changes.append(f"role={role}")
            except ValueError:
                pass
        if display_name is not None:
            user.display_name = display_name
            changes.append("display_name")
        if email is not None:
            user.email = email
            changes.append("email")
        if active is not None:
            user.is_active = active
            changes.append(f"active={active}")

        db.commit()
        log_audit(username=updated_by, action="user_update",
                   target_username=username,
                   message=f"Updated fields: {', '.join(changes)}",
                   ip_address=request.remote_addr if request else None)
        return True, f"کاربر '{username}' به‌روزرسانی شد"
    except Exception as exc:
        db.rollback()
        return False, f"خطا: {exc}"
    finally:
        db.close()


def delete_user(username: str, deleted_by: str = None) -> Tuple[bool, str]:
    """Delete a user. Cannot delete the last admin or self."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.username == username).first()
        if not user:
            return False, f"کاربر '{username}' یافت نشد"

        # Prevent deleting the last admin
        admin_count = db.query(models.User).filter(
            models.User.role == models.UserRole.ADMIN,
            models.User.is_active == True,
        ).count()
        if user.role == models.UserRole.ADMIN and admin_count <= 1:
            return False, "نمی‌توانید آخرین admin را حذف کنید"

        # Prevent self-deletion
        if username == deleted_by:
            return False, "نمی‌توانید حساب کاربری خودتان را حذف کنید"

        user_id_str = str(user.id)
        db.delete(user)
        db.commit()
        log_audit(username=deleted_by, action="user_delete",
                   target_username=username, target_resource=user_id_str,
                   message=f"User '{username}' deleted",
                   ip_address=request.remote_addr if request else None)
        return True, f"کاربر '{username}' حذف شد"
    except Exception as exc:
        db.rollback()
        return False, f"خطا: {exc}"
    finally:
        db.close()


# ── Session management ─────────────────────────────────────────────────────
def list_active_sessions() -> List[Dict[str, Any]]:
    """List all active sessions (for admin 'who's online' view)."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        sessions = db.query(models.UserSession).filter(
            models.UserSession.status == models.SessionStatus.ACTIVE
        ).order_by(models.UserSession.last_activity_at.desc()).all()
        return [{
            "id": str(s.id),
            "user_id": str(s.user_id),
            "username": s.user.username if s.user else None,
            "ip_address": s.ip_address,
            "user_agent": s.user_agent,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "last_activity_at": s.last_activity_at.isoformat() if s.last_activity_at else None,
            "expires_at": s.expires_at.isoformat() if s.expires_at else None,
        } for s in sessions]
    finally:
        db.close()


def revoke_session(session_id: str, revoked_by: str = None) -> Tuple[bool, str]:
    """Revoke (force-logout) a specific session. Admin only."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        from uuid import UUID
        s = db.query(models.UserSession).filter(models.UserSession.id == UUID(session_id)).first()
        if not s:
            return False, "Session not found"
        s.status = models.SessionStatus.REVOKED
        s.revoked_at = datetime.utcnow()
        s.revoked_by = revoked_by
        db.commit()
        log_audit(username=revoked_by, action="user_update",
                   target_resource=session_id,
                   message=f"Session revoked for user {s.user.username if s.user else '?'}",
                   ip_address=request.remote_addr if request else None)
        return True, "Session revoked"
    except Exception as exc:
        db.rollback()
        return False, f"خطا: {exc}"
    finally:
        db.close()


def revoke_all_user_sessions(user_id: str, revoked_by: str = None) -> int:
    """Revoke all active sessions for a user. Returns count revoked."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        from uuid import UUID
        sessions = db.query(models.UserSession).filter(
            models.UserSession.user_id == UUID(user_id),
            models.UserSession.status == models.SessionStatus.ACTIVE,
        ).all()
        for s in sessions:
            s.status = models.SessionStatus.REVOKED
            s.revoked_at = datetime.utcnow()
            s.revoked_by = revoked_by
        db.commit()
        return len(sessions)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to revoke sessions: %s", exc)
        return 0
    finally:
        db.close()


# ── Audit log query ────────────────────────────────────────────────────────
def list_audit_logs(limit: int = 100, username: str = None,
                    action: str = None, since: datetime = None) -> List[Dict[str, Any]]:
    """Query audit logs (admin only)."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        q = db.query(models.AuditLog)
        if username:
            q = q.filter(models.AuditLog.username == username)
        if action:
            try:
                action_enum = models.AuditAction(action)
                q = q.filter(models.AuditLog.action == action_enum)
            except ValueError:
                pass
        if since:
            q = q.filter(models.AuditLog.created_at >= since)
        entries = q.order_by(models.AuditLog.created_at.desc()).limit(limit).all()
        return [{
            "id": e.id,
            "user_id": str(e.user_id) if e.user_id else None,
            "username": e.username,
            "action": e.action.value if hasattr(e.action, "value") else str(e.action),
            "target_username": e.target_username,
            "target_resource": e.target_resource,
            "ip_address": e.ip_address,
            "success": e.success,
            "message": e.message,
            "details": e.details,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        } for e in entries]
    finally:
        db.close()


# ── Preset management (replaces preset_store.py) ───────────────────────────
def list_presets(user_id: str, include_admin_all: bool = True) -> List[Dict[str, Any]]:
    """
    List presets for a user. If include_admin_all and user is admin, returns
    ALL presets (across all users). Otherwise only the user's own presets.
    """
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        from uuid import UUID
        q = db.query(models.IOAssignmentPreset)
        # Check if user is admin
        user = db.query(models.User).filter(models.User.id == UUID(user_id)).first()
        if not (user and user.role == models.UserRole.ADMIN and include_admin_all):
            q = q.filter(models.IOAssignmentPreset.user_id == UUID(user_id))
        presets = q.order_by(models.IOAssignmentPreset.updated_at.desc()).all()
        return [p.to_dict(include_config=False) for p in presets]
    finally:
        db.close()


def get_preset(preset_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a single preset by ID. User must own it (or be admin)."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        from uuid import UUID
        preset = db.query(models.IOAssignmentPreset).filter(
            models.IOAssignmentPreset.id == UUID(preset_id)
        ).first()
        if not preset:
            return None
        # Check ownership
        user = db.query(models.User).filter(models.User.id == UUID(user_id)).first()
        if preset.user_id != UUID(user_id) and not (user and user.role == models.UserRole.ADMIN):
            return None
        return preset.to_dict(include_config=True)
    finally:
        db.close()


def upsert_preset(payload: Dict[str, Any], user_id: str,
                  username: str = None) -> Dict[str, Any]:
    """Create or update a preset by (user_id, project_name)."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        from uuid import UUID
        project_name = str(payload.get("project_name", "")).strip()
        if not project_name:
            raise ValueError("project_name is required")

        existing = db.query(models.IOAssignmentPreset).filter(
            models.IOAssignmentPreset.user_id == UUID(user_id),
            models.IOAssignmentPreset.project_name == project_name,
        ).first()

        if existing:
            existing.description = payload.get("description", "") or ""
            existing.type_count = int(payload.get("type_count", 1))
            existing.has_directions = bool(payload.get("has_directions", True))
            existing.cabinet_plan = payload.get("cabinet_plan", []) or []
            existing.cabinet_dimensions = payload.get("cabinet_dimensions", {}) or {}
            existing.card_catalog = payload.get("card_catalog", {}) or {}
            db.commit()
            log_audit(username=username, action="preset_update",
                       target_resource=str(existing.id),
                       target_username=project_name,
                       message=f"Preset '{project_name}' updated")
            return {"id": str(existing.id), "action": "updated", "project_name": project_name}
        else:
            preset = models.IOAssignmentPreset(
                user_id=UUID(user_id),
                project_name=project_name,
                description=payload.get("description", "") or "",
                type_count=int(payload.get("type_count", 1)),
                has_directions=bool(payload.get("has_directions", True)),
                cabinet_plan=payload.get("cabinet_plan", []) or [],
                cabinet_dimensions=payload.get("cabinet_dimensions", {}) or {},
                card_catalog=payload.get("card_catalog", {}) or {},
            )
            db.add(preset)
            db.commit()
            log_audit(username=username, action="preset_save",
                       target_resource=str(preset.id),
                       target_username=project_name,
                       message=f"Preset '{project_name}' created")
            return {"id": str(preset.id), "action": "created", "project_name": project_name}
    except Exception as exc:
        db.rollback()
        raise
    finally:
        db.close()


def delete_preset(preset_id: str, user_id: str, username: str = None) -> bool:
    """Delete a preset. User must own it (or be admin)."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        from uuid import UUID
        preset = db.query(models.IOAssignmentPreset).filter(
            models.IOAssignmentPreset.id == UUID(preset_id)
        ).first()
        if not preset:
            return False
        user = db.query(models.User).filter(models.User.id == UUID(user_id)).first()
        if preset.user_id != UUID(user_id) and not (user and user.role == models.UserRole.ADMIN):
            return False
        project_name = preset.project_name
        db.delete(preset)
        db.commit()
        log_audit(username=username, action="preset_delete",
                   target_resource=preset_id,
                   target_username=project_name,
                   message=f"Preset '{project_name}' deleted")
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


# ── CLI for seeding users ──────────────────────────────────────────────────
def seed_admin_if_empty(default_password: str = "admin123") -> None:
    """Ensure an admin user exists. Call this on app startup."""
    models = _get_models()
    SessionLocal = _get_db()
    db = SessionLocal()
    try:
        admin = db.query(models.User).filter(models.User.username == "admin").first()
        if admin:
            return
        admin = models.User(
            username="admin",
            role=models.UserRole.ADMIN,
            display_name="Administrator",
            email="",
            is_active=True,
            created_by="system",
        )
        admin.set_password(default_password)
        db.add(admin)
        db.commit()
        logger.info("Default admin user created (password: %s — change immediately!)", default_password)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to seed admin: %s", exc)
    finally:
        db.close()
