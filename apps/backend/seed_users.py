#!/usr/bin/env python3
"""
seed_users.py — Seed initial users into the database.

Usage:
    python3 seed_users.py                    # Creates default admin only (if missing)
    python3 seed_users.py --admin-password X # Creates admin with custom password
    python3 seed_users.py --from-json users.json  # Migrate users from JSON file

Run this once after applying the migration. The default admin password is
"admin123" — change it immediately after first login.
"""

import argparse
import json
import logging
import os
import sys

# Add parent dir to path so we can import the app's db modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Seed initial users")
    parser.add_argument("--admin-password", default="admin123",
                        help="Password for default admin user (default: admin123)")
    parser.add_argument("--from-json", default=None,
                        help="Path to users.json file to migrate from")
    parser.add_argument("--extra-users", action="store_true",
                        help="Also create sample engineer and viewer users")
    args = parser.parse_args()

    try:
        from user_manager_db import create_user, seed_admin_if_empty, get_user
    except ImportError as exc:
        logger.error("Cannot import user_manager_db: %s", exc)
        logger.error("Make sure you're running this from the same directory as user_manager_db.py")
        sys.exit(1)

    # 1) Ensure default admin exists
    logger.info("Checking for default admin user...")
    seed_admin_if_empty(default_password=args.admin_password)

    admin = get_user("admin")
    if admin:
        logger.info("✅ Admin user present: id=%s, role=%s, active=%s",
                     admin["id"], admin["role"], admin["is_active"])
    else:
        logger.error("❌ Admin user creation failed")
        sys.exit(1)

    # 2) Optionally migrate from JSON file
    if args.from_json:
        if not os.path.exists(args.from_json):
            logger.error("JSON file not found: %s", args.from_json)
            sys.exit(1)
        logger.info("Migrating users from %s ...", args.from_json)
        with open(args.from_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        migrated = 0
        for username, info in data.items():
            if username.startswith("_"):
                continue
            if not isinstance(info, dict) or "password" not in info:
                continue
            existing = get_user(username)
            if existing:
                logger.info("  Skipping '%s' (already exists)", username)
                continue
            success, msg = create_user(
                username=username,
                password=info["password"],
                role=info.get("role", "engineer"),
                display_name=info.get("display_name", username),
                email=info.get("email", ""),
                created_by="migration",
            )
            if success:
                migrated += 1
                logger.info("  ✅ Migrated '%s' (role=%s)", username, info.get("role", "engineer"))
            else:
                logger.warning("  ❌ Failed '%s': %s", username, msg)
        logger.info("Migration complete: %d users migrated", migrated)

    # 3) Optionally create sample users
    if args.extra_users:
        for username, password, role, display in [
            ("engineer1", "engineer123", "engineer", "Sample Engineer"),
            ("viewer1", "viewer123", "viewer", "Sample Viewer"),
        ]:
            existing = get_user(username)
            if existing:
                logger.info("Sample user '%s' already exists — skipping", username)
                continue
            success, msg = create_user(
                username=username,
                password=password,
                role=role,
                display_name=display,
                created_by="seed",
            )
            if success:
                logger.info("✅ Created sample user '%s' (role=%s, password=%s)",
                             username, role, password)
            else:
                logger.warning("❌ Failed to create '%s': %s", username, msg)

    logger.info("Done. Login with admin / %s", args.admin_password)


if __name__ == "__main__":
    main()
