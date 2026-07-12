"""
dimension_api.py — API endpoints for dimension-based cabinet configuration.

This module provides API endpoints that allow the frontend to:
1. Preview derived cabinet_limits from physical dimensions
2. Get fill data for cabinet visualization
3. Switch between dimension mode and percentage mode

CRITICAL: This module does NOT modify engine.py, app.py, or any existing file.
It's a NEW Flask Blueprint that can be registered alongside the existing api_bp.

Usage in app.py (ONE LINE to add):
    from apps.backend.modules.io_assignment.dimension_api import dimension_bp
    app.register_blueprint(dimension_bp)

That's the ONLY change to app.py — one import + one register_blueprint line.
"""

import logging
from flask import Blueprint, jsonify, request, session
from typing import Dict, Any

# Import our dimension calculator (NEW file, doesn't touch engine)
from apps.backend.modules.io_assignment.dimension_calculator import (
    derive_cabinet_limits,
    calculate_fill_data,
    CardCatalog,
    TerminalRailDimensions,
    CardColumnDimensions,
    BOARD_TYPE_KEYS,
)

dimension_bp = Blueprint("dimension_api", __name__, url_prefix="/api/dimensions")
logger = logging.getLogger(__name__)


def _require_auth():
    """Check if user is logged in. Returns username or None."""
    username = session.get("username")
    if not username:
        return None
    return str(username)


@dimension_bp.route("/preview", methods=["POST"])
def preview_dimensions():
    """
    POST /api/dimensions/preview
    
    Accepts physical dimensions and returns:
    - derived cabinet_limits (same format as engine's DEFAULT_CABINET_LIMITS)
    - fill data for UI visualization
    
    Request body (JSON):
    {
        "cabinet_dimensions": {
            "Type 1": {
                "FRONT": {
                    "terminal_rail": {
                        "rail_count": 4,
                        "rail_length_mm": 1711,
                        "top_gap_mm": 52,
                        "bottom_gap_mm": 20,
                        "terminal_height_mm": 5,
                        "jb_gap_mm": 10
                    },
                    "card_column": {
                        "column_count": 2,
                        "column_length_mm": 1711,
                        "top_gap_mm": 52,
                        "bottom_gap_mm": 20,
                        "reserved_space_mm": 300,
                        "card_gap_mm": 2
                    }
                },
                "REAR": { ... }
            }
        },
        "card_catalog": {
            "card_heights_mm": {
                "Barrier Board (AI)": 75,
                "Barrier Board (AO)": 75,
                ...
            }
        }
    }
    
    Response (JSON):
    {
        "status": "success",
        "cabinet_limits": {
            "Type 1|FRONT": {
                "Max_Total_Boards": 100,
                "Max rail terminals": 1308,
                "Barrier Board (AI)": 2.9,
                ...
            },
            ...
        },
        "fill_data": {
            "Type 1": {
                "FRONT": {
                    "terminal_rails": [...],
                    "card_columns": [...],
                    "max_terminals": 1308,
                    "derived_limits": {...}
                }
            }
        }
    }
    """
    username = _require_auth()
    if not username:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        data = request.get_json(force=True)
        cabinet_dimensions = data.get("cabinet_dimensions", {})
        card_catalog_data = data.get("card_catalog", {})

        # Build CardCatalog from request
        card_catalog = CardCatalog()
        if card_catalog_data and "card_heights_mm" in card_catalog_data:
            card_catalog.card_heights_mm.update(card_catalog_data["card_heights_mm"])

        # Derive cabinet limits
        cabinet_limits = derive_cabinet_limits(cabinet_dimensions, card_catalog)

        # Calculate fill data for visualization
        fill_data = calculate_fill_data(cabinet_dimensions, card_catalog)

        return jsonify({
            "status": "success",
            "cabinet_limits": cabinet_limits,
            "fill_data": fill_data
        })

    except Exception as e:
        logger.error(f"Error in preview_dimensions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@dimension_bp.route("/reverse-engineer", methods=["POST"])
def reverse_engineer():
    """
    POST /api/dimensions/reverse-engineer
    
    Given current percentages, reverse-engineer the card dimensions that
    would produce them. Used to pre-populate the dimension UI with
    reasonable defaults based on the current percentage values.
    
    Request body (JSON):
    {
        "percentages": {
            "Barrier Board (AI)": 14.2,
            "Barrier Board (AO)": 14.2,
            ...
        },
        "rail_length_mm": 1711,
        "top_gap_mm": 52,
        "bottom_gap_mm": 20,
        "reserved_space_mm": 0,
        "card_gap_mm": 2,
        "column_count": 1
    }
    
    Response:
    {
        "status": "success",
        "card_dimensions": {
            "Barrier Board (AI)": {
                "card_height_mm": 232.1,
                "cards_fit": 7,
                "total_cards": 7
            },
            ...
        }
    }
    """
    username = _require_auth()
    if not username:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        from apps.backend.modules.io_assignment.dimension_calculator import (
            reverse_engineer_dimensions
        )

        data = request.get_json(force=True)
        percentages = data.get("percentages", {})
        rail_length = data.get("rail_length_mm", 1711)
        top_gap = data.get("top_gap_mm", 52)
        bottom_gap = data.get("bottom_gap_mm", 20)
        reserved = data.get("reserved_space_mm", 0)
        card_gap = data.get("card_gap_mm", 2)
        columns = data.get("column_count", 1)

        result = {}
        for board_key, pct in percentages.items():
            if pct and pct > 0:
                result[board_key] = reverse_engineer_dimensions(
                    float(pct), rail_length, top_gap, bottom_gap,
                    reserved, card_gap, columns
                )

        return jsonify({
            "status": "success",
            "card_dimensions": result
        })

    except Exception as e:
        logger.error(f"Error in reverse_engineer: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@dimension_bp.route("/defaults", methods=["GET"])
def get_default_dimensions():
    """
    GET /api/dimensions/defaults
    
    Returns default dimension values that reproduce the current
    DEFAULT_CABINET_LIMITS percentages. This lets the UI pre-populate
    the dimension fields with sensible defaults.
    
    Response:
    {
        "status": "success",
        "defaults": {
            "terminal_rail": {
                "rail_count": 4,
                "rail_length_mm": 1711,
                "top_gap_mm": 52,
                "bottom_gap_mm": 20,
                "terminal_height_mm": 5,
                "jb_gap_mm": 10
            },
            "card_column": {
                "column_count": 1,
                "column_length_mm": 1711,
                "top_gap_mm": 52,
                "bottom_gap_mm": 20,
                "reserved_space_mm": 0,
                "card_gap_mm": 2
            },
            "card_heights_mm": {
                "Barrier Board (AI)": 75,
                ...
            }
        }
    }
    """
    username = _require_auth()
    if not username:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        # Default card catalog
        catalog = CardCatalog()

        return jsonify({
            "status": "success",
            "defaults": {
                "terminal_rail": {
                    "rail_count": 4,
                    "rail_length_mm": 1711,
                    "top_gap_mm": 52,
                    "bottom_gap_mm": 20,
                    "terminal_height_mm": 5,
                    "jb_gap_mm": 10
                },
                "card_column": {
                    "column_count": 1,
                    "column_length_mm": 1711,
                    "top_gap_mm": 52,
                    "bottom_gap_mm": 20,
                    "reserved_space_mm": 0,
                    "card_gap_mm": 2
                },
                "card_heights_mm": catalog.card_heights_mm
            }
        })

    except Exception as e:
        logger.error(f"Error in get_default_dimensions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
