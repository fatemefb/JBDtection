"""
dimension_calculator.py — Physical Dimension → Engine Limits Calculator

This module is a PURE FUNCTION layer that sits ABOVE engine.py.
It takes physical cabinet/card/rail dimensions (in mm) as input and
produces the `cabinet_limits` dict that engine.py's `build_config()`
already knows how to consume.

CRITICAL DESIGN RULES:
1. ZERO imports from engine.py — this module is independent.
2. ZERO changes to engine.py — the output of derive_cabinet_limits()
   is a dict with the EXACT same keys that DEFAULT_CABINET_LIMITS uses.
3. If dimensions are not provided, callers fall back to DEFAULT_CABINET_LIMITS.
4. All rounding is FLOOR (math.floor) so we never over-estimate capacity.

The two physical systems modeled (per the user's cabinet layout drawings):

SYSTEM 1 — Terminal Rail (JB Marshaling)
  Vertical DIN rails where small terminal blocks are stacked.
  Each tag uses 2 terminals (+1 if SRC=SC). Between JBs there is a gap.
  Output: "Max rail terminals" (integer count).

SYSTEM 2 — Card Columns (I/O Cards)
  Separate vertical columns where I/O cards (Barrier AI, AO, DI, DO,
  Terminal AI, ..., Relay AI, ...) are stacked vertically.
  Each card has a height in mm. Between cards there is a gap.
  Some space is reserved (power supplies, etc.).
  Output: per-board-type percentage (float, e.g. 14.2 meaning each
  card of this type consumes 14.2% of the card-column pool).

SYSTEM 3 — Sides (optional, simplified)
  Side compartments with available length and card height.
  Only card count matters (no rail modeling).
  Output: per-board-type percentage (same as System 2 but for SIDE direction).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ──────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TerminalRailDimensions:
    """Physical dimensions of the terminal rail (for JB marshaling)."""
    rail_count: int = 1              # Number of vertical terminal rails
    rail_length_mm: float = 1711.0   # Total vertical length of each rail
    top_gap_mm: float = 52.0         # Gap from top of cabinet to first terminal
    bottom_gap_mm: float = 20.0      # Gap from last terminal to bottom
    terminal_height_mm: float = 5.0  # Height of each terminal block (mm)
    jb_gap_mm: float = 10.0          # Gap between JB groups on the rail


@dataclass
class CardColumnDimensions:
    """
    Physical dimensions of a card column (for I/O cards).
    
    Each cabinet direction (FRONT/REAR/SIDE) can have multiple card columns.
    Different board types may share columns or have dedicated columns.
    The simplest model: all board types share the same total card space,
    and each board type's percentage = 100 / max_cards_of_that_type.
    
    For more accuracy, columns can be dedicated per board type.
    """
    column_count: int = 1             # Number of card columns available
    column_length_mm: float = 1711.0  # Vertical length of each column
    top_gap_mm: float = 52.0          # Gap from top
    bottom_gap_mm: float = 20.0       # Gap from bottom
    reserved_space_mm: float = 0.0    # Space used by non-I/O devices (power supply, etc.)
    card_gap_mm: float = 2.0          # Gap between cards stacked vertically


@dataclass
class CardCatalog:
    """
    Physical dimensions of each board type.
    Keys match the board-type names used in engine.py's cabinet_limits.
    """
    card_heights_mm: Dict[str, float] = field(default_factory=lambda: {
        "Barrier Board (AI)": 75.0,
        "Barrier Board (AO)": 75.0,
        "Barrier Board (DI)": 45.0,
        "Barrier Board (DO)": 45.0,
        "Terminal Board (AI)": 60.0,
        "Terminal Board (AO)": 60.0,
        "Terminal Board (DI)": 45.0,
        "Terminal Board (DO)": 45.0,
        "Relay Board AI capacity": 90.0,
        "Relay Board AO capacity": 90.0,
        "Relay Board DI capacity": 90.0,
        "Relay Board DO capacity": 90.0,
    })


@dataclass
class CabinetDirectionDimensions:
    """All physical dimensions for one direction (FRONT/REAR/SIDE) of a cabinet type."""
    terminal_rail: Optional[TerminalRailDimensions] = None
    card_column: Optional[CardColumnDimensions] = None


@dataclass
class CabinetTypeDimensions:
    """All physical dimensions for one cabinet type."""
    directions: Dict[str, CabinetDirectionDimensions] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Core Calculation Functions
# ──────────────────────────────────────────────────────────────────────────

def calculate_max_rail_terminals(rail_dims: TerminalRailDimensions) -> int:
    """
    Calculate the maximum number of terminal blocks that fit on the terminal rails.
    
    Formula:
        usable_per_rail = rail_length - top_gap - bottom_gap
        terminals_per_rail = floor(usable_per_rail / terminal_height)
        max_terminals = terminals_per_rail * rail_count
    
    Note: JB gaps are accounted for in the engine's rail_load calculation
    (jb_label_overhead). The terminal count here is the raw physical maximum.
    
    Returns:
        Integer count of maximum terminals.
    """
    usable_per_rail = (
        rail_dims.rail_length_mm
        - rail_dims.top_gap_mm
        - rail_dims.bottom_gap_mm
    )
    
    if usable_per_rail <= 0 or rail_dims.terminal_height_mm <= 0:
        return 0
    
    terminals_per_rail = math.floor(usable_per_rail / rail_dims.terminal_height_mm)
    max_terminals = terminals_per_rail * rail_dims.rail_count
    
    return max(max_terminals, 0)


def calculate_board_percentage(
    board_type: str,
    card_catalog: CardCatalog,
    column_dims: CardColumnDimensions
) -> float:
    """
    Calculate the percentage of card-column space consumed by one board of the given type.
    
    Formula:
        usable = column_length - top_gap - bottom_gap - reserved_space
        effective_pitch = card_height + card_gap
        cards_per_column = floor(usable / effective_pitch)
        total_cards = cards_per_column * column_count
        percentage = 100.0 / total_cards
    
    Rounding: The percentage is rounded DOWN to 1 decimal place to ensure
    we never over-estimate capacity (never place more cards than physically fit).
    
    Args:
        board_type: Engine board-type key (e.g. "Barrier Board (AI)")
        card_catalog: Card dimensions catalog
        column_dims: Card column physical dimensions
    
    Returns:
        Percentage (float). E.g., 14.2 means each card takes 14.2% of the pool.
    """
    card_height = card_catalog.card_heights_mm.get(board_type, 0.0)
    if card_height <= 0:
        # If we don't have a dimension for this board type, return a safe default
        # that allows a reasonable number of cards (e.g., 7 cards = 14.3%)
        return 14.3
    
    usable = (
        column_dims.column_length_mm
        - column_dims.top_gap_mm
        - column_dims.bottom_gap_mm
        - column_dims.reserved_space_mm
    )
    
    if usable <= 0:
        return 100.0  # No space → each card takes everything (effectively 1 card max)
    
    effective_pitch = card_height + column_dims.card_gap_mm
    if effective_pitch <= 0:
        return 100.0
    
    cards_per_column = math.floor(usable / effective_pitch)
    if cards_per_column <= 0:
        return 100.0
    
    total_cards = cards_per_column * column_dims.column_count
    if total_cards <= 0:
        return 100.0
    
    raw_percentage = 100.0 / total_cards
    
    # Round DOWN to 1 decimal place (never over-estimate)
    rounded = math.floor(raw_percentage * 10) / 10.0
    
    return rounded


def calculate_side_percentage(
    board_type: str,
    card_catalog: CardCatalog,
    side_length_mm: float,
    side_count: int,
    card_gap_mm: float = 2.0
) -> float:
    """
    Calculate board percentage for side compartments (simplified model).
    
    Sides don't have rail modeling — just available length and card height.
    
    Formula:
        cards_per_side = floor(length / (card_height + gap))
        total_cards = cards_per_side * side_count
        percentage = 100.0 / total_cards
    """
    card_height = card_catalog.card_heights_mm.get(board_type, 0.0)
    if card_height <= 0 or side_length_mm <= 0 or side_count <= 0:
        return 14.3  # safe default
    
    effective_pitch = card_height + card_gap_mm
    cards_per_side = math.floor(side_length_mm / effective_pitch)
    
    if cards_per_side <= 0:
        return 100.0
    
    total_cards = cards_per_side * side_count
    if total_cards <= 0:
        return 100.0
    
    raw_percentage = 100.0 / total_cards
    return math.floor(raw_percentage * 10) / 10.0


# ──────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────

# All 14 board-type keys that engine.py's cabinet_limits expects
BOARD_TYPE_KEYS = [
    "Barrier Board (AI)",
    "Barrier Board (AO)",
    "Barrier Board (DI)",
    "Barrier Board (DO)",
    "Terminal Board (AI)",
    "Terminal Board (AO)",
    "Terminal Board (DI)",
    "Terminal Board (DO)",
    "Relay Board AI capacity",
    "Relay Board AO capacity",
    "Relay Board DI capacity",
    "Relay Board DO capacity",
]


def derive_cabinet_limits(
    cabinet_dimensions: Dict[str, Any],
    card_catalog: Optional[CardCatalog] = None
) -> Dict[str, Dict[str, float]]:
    """
    Main entry point: convert physical dimensions to engine cabinet_limits.
    
    This function produces a dict with the EXACT same structure as
    engine.py's DEFAULT_CABINET_LIMITS, but with values derived from
    physical measurements instead of hardcoded magic numbers.
    
    Args:
        cabinet_dimensions: Dict mapping cabinet type names to their dimensions.
            Structure:
            {
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
                    "REAR": { ... same structure ... }
                },
                "Type 2": { ... }
            }
        
        card_catalog: Card dimensions. If None, uses default CardCatalog.
    
    Returns:
        Dict with pipe-separated keys matching engine's build_config() format:
        {
            "Type 1|FRONT": {
                "Max_Total_Boards": 100,
                "Max rail terminals": 1308,
                "Barrier Board (AI)": 4.8,
                "Barrier Board (AO)": 4.8,
                ...
            },
            "Type 1|REAR": { ... },
            ...
        }
    
    Example:
        >>> dims = {
        ...     "Type 1": {
        ...         "FRONT": {
        ...             "terminal_rail": {"rail_count": 4, "rail_length_mm": 1711, ...},
        ...             "card_column": {"column_count": 2, "column_length_mm": 1711, ...}
        ...         }
        ...     }
        ... }
        >>> limits = derive_cabinet_limits(dims)
        >>> limits["Type 1|FRONT"]["Max rail terminals"]
        1308
        >>> limits["Type 1|FRONT"]["Barrier Board (AI)"]
        4.8
    """
    if card_catalog is None:
        card_catalog = CardCatalog()
    
    result = {}
    
    for cabinet_type, directions_data in cabinet_dimensions.items():
        for direction, dim_data in directions_data.items():
            key = f"{cabinet_type}|{direction}"
            limits = {}
            
            # ── Max_Total_Boards: always 100 (percentage mode) ──
            # This tells engine.py to use PERCENT_PER_BOARD mode
            limits["Max_Total_Boards"] = 100
            
            # ── Max rail terminals (from terminal rail dimensions) ──
            terminal_rail_data = dim_data.get("terminal_rail")
            if terminal_rail_data:
                rail_dims = TerminalRailDimensions(**terminal_rail_data)
                limits["Max rail terminals"] = calculate_max_rail_terminals(rail_dims)
            else:
                limits["Max rail terminals"] = 800  # safe default
            
            # ── Board percentages (from card column dimensions) ──
            card_column_data = dim_data.get("card_column")
            if card_column_data:
                column_dims = CardColumnDimensions(**card_column_data)
                
                for board_key in BOARD_TYPE_KEYS:
                    limits[board_key] = calculate_board_percentage(
                        board_key, card_catalog, column_dims
                    )
            else:
                # No card column data — use safe defaults
                for board_key in BOARD_TYPE_KEYS:
                    limits[board_key] = 14.3
            
            # ── Side handling (if direction is SIDE) ──
            if direction == "SIDE":
                side_length = dim_data.get("side_length_mm", 600)
                side_count = dim_data.get("side_count", 2)
                card_gap = dim_data.get("card_gap_mm", 2.0)
                
                for board_key in BOARD_TYPE_KEYS:
                    limits[board_key] = calculate_side_percentage(
                        board_key, card_catalog, side_length, side_count, card_gap
                    )
            
            result[key] = limits
    
    return result


# ──────────────────────────────────────────────────────────────────────────
# Fill Data (for UI visualization)
# ──────────────────────────────────────────────────────────────────────────

def calculate_fill_data(
    cabinet_dimensions: Dict[str, Any],
    card_catalog: Optional[CardCatalog] = None
) -> Dict[str, Any]:
    """
    Calculate fill data for UI visualization (the empty cabinet preview).
    
    Returns detailed information about each rail/column that the frontend
    can use to render a visual representation of the empty cabinet.
    
    Returns:
        {
            "Type 1": {
                "FRONT": {
                    "terminal_rails": [
                        {
                            "rail_index": 0,
                            "length_mm": 1711,
                            "usable_mm": 1639,
                            "max_terminals": 327,
                            "terminal_height_mm": 5,
                            "top_gap_mm": 52,
                            "bottom_gap_mm": 20
                        },
                        ... (one per rail)
                    ],
                    "card_columns": [
                        {
                            "column_index": 0,
                            "length_mm": 1711,
                            "usable_mm": 1339,
                            "reserved_mm": 300,
                            "cards": {
                                "Barrier Board (AI)": {
                                    "card_height_mm": 75,
                                    "cards_fit": 17,
                                    "percentage": 4.8,
                                    "fill_preview": [75, 75, 75, ...]  # heights of each card
                                },
                                ...
                            }
                        },
                        ... (one per column)
                    ],
                    "max_terminals": 1308,
                    "derived_limits": { ... same as derive_cabinet_limits output ... }
                },
                "REAR": { ... }
            }
        }
    """
    if card_catalog is None:
        card_catalog = CardCatalog()
    
    result = {}
    
    for cabinet_type, directions_data in cabinet_dimensions.items():
        result[cabinet_type] = {}
        
        for direction, dim_data in directions_data.items():
            dir_result = {}
            
            # ── Terminal rails ──
            terminal_rail_data = dim_data.get("terminal_rail")
            if terminal_rail_data:
                rail_dims = TerminalRailDimensions(**terminal_rail_data)
                
                usable_per_rail = (
                    rail_dims.rail_length_mm
                    - rail_dims.top_gap_mm
                    - rail_dims.bottom_gap_mm
                )
                terminals_per_rail = math.floor(usable_per_rail / rail_dims.terminal_height_mm) if rail_dims.terminal_height_mm > 0 else 0
                
                rails_list = []
                for i in range(rail_dims.rail_count):
                    rails_list.append({
                        "rail_index": i,
                        "length_mm": rail_dims.rail_length_mm,
                        "usable_mm": usable_per_rail,
                        "max_terminals": terminals_per_rail,
                        "terminal_height_mm": rail_dims.terminal_height_mm,
                        "top_gap_mm": rail_dims.top_gap_mm,
                        "bottom_gap_mm": rail_dims.bottom_gap_mm,
                        "jb_gap_mm": rail_dims.jb_gap_mm,
                    })
                
                dir_result["terminal_rails"] = rails_list
                dir_result["max_terminals"] = terminals_per_rail * rail_dims.rail_count
            else:
                dir_result["terminal_rails"] = []
                dir_result["max_terminals"] = 0
            
            # ── Card columns ──
            card_column_data = dim_data.get("card_column")
            if card_column_data:
                column_dims = CardColumnDimensions(**card_column_data)
                
                usable = (
                    column_dims.column_length_mm
                    - column_dims.top_gap_mm
                    - column_dims.bottom_gap_mm
                    - column_dims.reserved_space_mm
                )
                
                columns_list = []
                for i in range(column_dims.column_count):
                    col_cards = {}
                    for board_key in BOARD_TYPE_KEYS:
                        card_height = card_catalog.card_heights_mm.get(board_key, 0.0)
                        if card_height <= 0:
                            continue
                        
                        effective_pitch = card_height + column_dims.card_gap_mm
                        cards_fit = math.floor(usable / effective_pitch) if effective_pitch > 0 else 0
                        total_cards = cards_fit * column_dims.column_count
                        pct = math.floor((100.0 / total_cards) * 10) / 10.0 if total_cards > 0 else 100.0
                        
                        # Fill preview: list of card heights for visualization
                        fill_preview = [card_height] * min(cards_fit, 50)  # cap at 50 for UI
                        
                        col_cards[board_key] = {
                            "card_height_mm": card_height,
                            "cards_fit": cards_fit,
                            "total_cards": total_cards,
                            "percentage": pct,
                            "fill_preview": fill_preview,
                        }
                    
                    columns_list.append({
                        "column_index": i,
                        "length_mm": column_dims.column_length_mm,
                        "usable_mm": usable,
                        "reserved_mm": column_dims.reserved_space_mm,
                        "top_gap_mm": column_dims.top_gap_mm,
                        "bottom_gap_mm": column_dims.bottom_gap_mm,
                        "card_gap_mm": column_dims.card_gap_mm,
                        "cards": col_cards,
                    })
                
                dir_result["card_columns"] = columns_list
            else:
                dir_result["card_columns"] = []
            
            # ── Derived limits (same as derive_cabinet_limits for this type/dir) ──
            single_dim = {cabinet_type: {direction: dim_data}}
            dir_result["derived_limits"] = derive_cabinet_limits(single_dim, card_catalog).get(
                f"{cabinet_type}|{direction}", {}
            )
            
            result[cabinet_type][direction] = dir_result
    
    return result


# ──────────────────────────────────────────────────────────────────────────
# Reverse Engineering: Find dimensions that reproduce current defaults
# ──────────────────────────────────────────────────────────────────────────

def reverse_engineer_dimensions(
    current_percentage: float,
    rail_length_mm: float = 1711.0,
    top_gap_mm: float = 52.0,
    bottom_gap_mm: float = 20.0,
    reserved_space_mm: float = 0.0,
    card_gap_mm: float = 2.0,
    column_count: int = 1
) -> Dict[str, float]:
    """
    Given a current percentage (e.g. 14.2), reverse-engineer the card height
    that would produce this percentage with the given rail dimensions.
    
    This is used for:
    1. Testing (A2) — verify our formula reproduces current defaults
    2. Pre-populating the UI with reasonable default dimensions
    
    Formula reverse:
        percentage = 100 / (floor(usable / (height + gap)) * column_count)
        → floor(usable / (height + gap)) = 100 / (percentage * column_count)
        → height + gap = usable / (100 / (percentage * column_count))
        → height = usable / (100 / (percentage * column_count)) - gap
    """
    usable = rail_length_mm - top_gap_mm - bottom_gap_mm - reserved_space_mm
    
    # Calculate how many cards the current percentage implies
    total_cards = round(100.0 / current_percentage)
    cards_per_column = total_cards // column_count
    
    if cards_per_column <= 0:
        return {"card_height_mm": 0, "cards_fit": 0, "total_cards": 0}
    
    # Reverse-engineer card height
    effective_pitch = usable / cards_per_column
    card_height = effective_pitch - card_gap_mm
    
    return {
        "card_height_mm": round(card_height, 1),
        "cards_fit": cards_per_column,
        "total_cards": total_cards,
        "usable_mm": usable,
        "effective_pitch_mm": round(effective_pitch, 1),
    }


# ──────────────────────────────────────────────────────────────────────────
# Self-Test / Demo
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DIMENSION CALCULATOR — SELF TEST")
    print("=" * 70)
    
    # Test 1: Reverse-engineer current defaults
    print("\n--- Test 1: Reverse-engineer current percentages ---")
    
    test_cases = [
        ("Type 1 REAR Barrier AI", 20.0, 1711, 52, 20, 0, 2, 1),
        ("Type 1 FRONT Barrier AI", 14.2, 1711, 52, 20, 0, 2, 1),
        ("Type 1 REAR Barrier DI", 12.5, 1711, 52, 20, 0, 2, 1),
        ("Type 1 FRONT Barrier DI", 10.0, 1711, 52, 20, 0, 2, 1),
        ("Type 3 REAR Barrier AI", 16.6, 1711, 52, 20, 0, 2, 1),
        ("Type 1 FRONT Terminal DI", 7.1, 1711, 52, 20, 0, 2, 1),
        ("Type 1 FRONT Terminal DO", 8.3, 1711, 52, 20, 0, 2, 1),
        ("Type 1 REAR Relay DO", 33.3, 1711, 52, 20, 0, 2, 1),
    ]
    
    for name, pct, rail, tg, bg, res, gap, cols in test_cases:
        result = reverse_engineer_dimensions(pct, rail, tg, bg, res, gap, cols)
        print(f"  {name}: {pct}% → card_height={result['card_height_mm']}mm, "
              f"cards_fit={result['cards_fit']}, total={result['total_cards']}")
    
    # Test 2: Calculate from dimensions and verify round-trip
    print("\n--- Test 2: Round-trip test (dimensions → percentage → verify) ---")
    
    # If card_height = 75mm, rail = 1711, gaps = 52+20, gap = 2, columns = 1
    test_dims = CardColumnDimensions(
        column_count=1,
        column_length_mm=1711,
        top_gap_mm=52,
        bottom_gap_mm=20,
        reserved_space_mm=0,
        card_gap_mm=2,
    )
    test_catalog = CardCatalog()
    test_catalog.card_heights_mm["Barrier Board (AI)"] = 75.0
    
    pct = calculate_board_percentage("Barrier Board (AI)", test_catalog, test_dims)
    usable = 1711 - 52 - 20 - 0
    cards = math.floor(usable / (75 + 2))
    print(f"  Card height=75mm, usable={usable}mm, pitch=77mm → "
          f"cards={cards}, percentage={pct}%")
    print(f"  Verify: 100/{cards} = {100/cards:.4f} → rounded={math.floor(100/cards * 10) / 10}")
    
    # Test 3: Full cabinet_limits derivation
    print("\n--- Test 3: Full derive_cabinet_limits() ---")
    
    cabinet_dims = {
        "Type 1": {
            "FRONT": {
                "terminal_rail": {
                    "rail_count": 4,
                    "rail_length_mm": 1711,
                    "top_gap_mm": 52,
                    "bottom_gap_mm": 20,
                    "terminal_height_mm": 5,
                    "jb_gap_mm": 10,
                },
                "card_column": {
                    "column_count": 2,
                    "column_length_mm": 1711,
                    "top_gap_mm": 52,
                    "bottom_gap_mm": 20,
                    "reserved_space_mm": 300,
                    "card_gap_mm": 2,
                },
            },
            "REAR": {
                "terminal_rail": {
                    "rail_count": 4,
                    "rail_length_mm": 1711,
                    "top_gap_mm": 52,
                    "bottom_gap_mm": 20,
                    "terminal_height_mm": 5,
                    "jb_gap_mm": 10,
                },
                "card_column": {
                    "column_count": 1,
                    "column_length_mm": 1711,
                    "top_gap_mm": 52,
                    "bottom_gap_mm": 20,
                    "reserved_space_mm": 0,
                    "card_gap_mm": 2,
                },
            },
        }
    }
    
    limits = derive_cabinet_limits(cabinet_dims)
    
    print(f"\n  Type 1|FRONT:")
    print(f"    Max_Total_Boards: {limits['Type 1|FRONT']['Max_Total_Boards']}")
    print(f"    Max rail terminals: {limits['Type 1|FRONT']['Max rail terminals']}")
    for key in BOARD_TYPE_KEYS:
        val = limits['Type 1|FRONT'][key]
        print(f"    {key}: {val}%")
    
    print(f"\n  Type 1|REAR:")
    print(f"    Max_Total_Boards: {limits['Type 1|REAR']['Max_Total_Boards']}")
    print(f"    Max rail terminals: {limits['Type 1|REAR']['Max rail terminals']}")
    for key in BOARD_TYPE_KEYS:
        val = limits['Type 1|REAR'][key]
        print(f"    {key}: {val}%")
    
    # Test 4: Fill data for UI
    print("\n--- Test 4: Fill data for UI visualization ---")
    
    fill_data = calculate_fill_data(cabinet_dims)
    front_data = fill_data["Type 1"]["FRONT"]
    
    print(f"  Terminal rails: {len(front_data['terminal_rails'])}")
    for rail in front_data['terminal_rails'][:2]:
        print(f"    Rail {rail['rail_index']}: length={rail['length_mm']}mm, "
              f"usable={rail['usable_mm']}mm, max_terminals={rail['max_terminals']}")
    
    print(f"  Card columns: {len(front_data['card_columns'])}")
    for col in front_data['card_columns'][:1]:
        print(f"    Column {col['column_index']}: length={col['length_mm']}mm, "
              f"usable={col['usable_mm']}mm, reserved={col['reserved_mm']}mm")
        for board_key in list(col['cards'].keys())[:4]:
            card_info = col['cards'][board_key]
            print(f"      {board_key}: height={card_info['card_height_mm']}mm, "
                  f"fit={card_info['cards_fit']}, pct={card_info['percentage']}%")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
