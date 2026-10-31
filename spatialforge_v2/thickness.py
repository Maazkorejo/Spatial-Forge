"""
SpatialForge CAD Thickness Engine: Callout Extraction, Layer Filtering & Category Estimation.
Deterministic priority order:
1. Drawing Callout (THK, T=, PL, DEPTH) -> source: "callout" (confidence: 0.95)
2. Multi-view / 2-view geometry        -> source: "measured" (confidence: 0.85 - 0.95)
3. Category-aware single-view fallback  -> source: "estimated" (confidence: <= 0.50)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Text / Annotation Thickness Patterns
# ---------------------------------------------------------------------------
FRACTION_PATTERNS = [
    # 1/4" PLATE, 3/8 INCH PL, 1/2 THK
    r'(\d+)\s*/\s*(\d+)\s*(?:"|IN|INCH|INCHES)?\s*(?:PLATE|PL|THK|THICK|THICKNESS)\b',
    # PLATE: 1/4", THK: 3/8 INCH
    r'(?:PLATE|PL|THK|THICK|THICKNESS)\s*[:=]?\s*(\d+)\s*/\s*(\d+)\s*(?:"|IN|INCH|INCHES)?',
]

DECIMAL_PATTERNS = [
    # Explicit thickness keywords with optional unit: THK 3.0, THICKNESS=5mm, THK: 0.25", THK 12
    r'(?<!/)\b(?:THK|THICK|THICKNESS)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(MM|IN|INCH|INCHES|")?(?!/)',
    # T=5 or T: 3.2mm or T 6.0
    r'(?<!/)\bT\s*[:=]\s*(\d+(?:\.\d+)?)\s*(MM|IN|INCH|INCHES|")?(?!/)',
    # Plate thickness: 10mm PLATE, PL: 6, PL 1/4"
    r'(?<!/)\bPL(?:ATE)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(MM|IN|INCH|INCHES|")?(?!/)',
    r'(?<!/)\b(\d+(?:\.\d+)?)\s*(MM|IN|INCH|INCHES|")?\s*PL(?:ATE)?\b',
    # Depth callouts: DEPTH 25, DEPTH=12mm
    r'(?<!/)\bDEPTH\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(MM|IN|INCH|INCHES|")?(?!/)',
]

def extract_thickness_callout(text_entities: List[str], scale_hint: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Search text notes, dimensions, and callouts for explicit material thickness callouts.
    Converts imperial inches and fractions to millimeters.
    """
    for raw in text_entities:
        if not raw:
            continue
        t = str(raw).upper().strip()
        
        # 1. Try fractional patterns first (e.g. 1/4" PLATE, 3/8 PL)
        for pat in FRACTION_PATTERNS:
            m = re.search(pat, t)
            if m:
                num, den = float(m.group(1)), float(m.group(2))
                if den > 0:
                    value_mm = (num / den) * 25.4
                    return {
                        "value_mm": round(value_mm, 2),
                        "source": "callout",
                        "confidence": 0.95,
                        "raw_text": raw.strip(),
                    }

        # 2. Try decimal / integer patterns (e.g. THK 3.0mm, T=5, PL: 8)
        for pat in DECIMAL_PATTERNS:
            m = re.search(pat, t)
            if m:
                value_raw = float(m.group(1))
                unit = (m.group(2) or "").upper() if len(m.groups()) >= 2 and m.group(2) else ""
                if unit in ("IN", "INCH", "INCHES", '"'):
                    value_mm = value_raw * 25.4
                else:
                    value_mm = value_raw
                return {
                    "value_mm": round(value_mm, 2),
                    "source": "callout",
                    "confidence": 0.95,
                    "raw_text": raw.strip(),
                }
    return None


# ---------------------------------------------------------------------------
# 2. Dimension Line / Annotation Layer Filtering
# ---------------------------------------------------------------------------
DIMENSION_LAYER_PATTERNS = re.compile(r'DIM|LEADER|ANNO|EXTENSION|DEFPOINTS|NOTATION', re.IGNORECASE)

def filter_geometry_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Exclude dimension/annotation entities before bounding-box clustering.
    Prevents extension lines and leader arrows from contaminating real geometry measurements.
    """
    clean = []
    for e in entities:
        lay = str(e.get("layer", "") or "")
        kind = str(e.get("kind", "") or "").lower()
        role = str(e.get("role", "") or "").lower()
        if DIMENSION_LAYER_PATTERNS.search(lay):
            continue
        if kind in ("dimension", "leader", "text", "mtext"):
            continue
        if role in ("dim", "title"):
            continue
        clean.append(e)
    return clean


# ---------------------------------------------------------------------------
# 3. Category-Aware Single-View Fallback
# ---------------------------------------------------------------------------
THICKNESS_CATEGORY_RANGES = {
    "sheet_metal": (2.0, 4.0, 0.03),   # (min_mm, max_mm, typical_fraction_of_min_dim)
    "hand_tool":   (6.0, 10.0, 0.07),
    "housing":     (25.0, 50.0, 0.32),
    "default":     (8.0, 20.0, 0.15),  # unknown category — general flat fallback
}

def estimate_thickness_single_view(
    w_mm: float,
    h_mm: float,
    category: str = "default",
    category_confidence: float = 0.0
) -> Dict[str, Any]:
    """
    Single-view estimation when zero depth geometry exists.
    Always tagged with source: 'estimated' and confidence capped <= 0.50.
    """
    lo, hi, frac = THICKNESS_CATEGORY_RANGES.get(category, THICKNESS_CATEGORY_RANGES["default"])
    min_span = max(1.0, min(w_mm, h_mm))
    est = min_span * frac
    est = max(lo, min(hi, est))
    return {
        "value_mm": round(est, 1),
        "source": "estimated",
        "category": category,
        "confidence": min(0.50, round(category_confidence, 2)),
    }


def map_family_to_thickness_category(family: str, title: str = "") -> str:
    """
    Map from the unified classification signal to a thickness category.
    Prevents dual-classifier drift.
    """
    text = f"{family} {title}".lower()
    if any(k in text for k in ("bracket", "butterfly", "plate", "sheet", "gasket", "panel", "flange", "disc", "cat")):
        return "sheet_metal"
    if any(k in text for k in ("tool", "wrench", "spanner", "pliers", "key", "handle", "arm", "lever")):
        return "hand_tool"
    if any(k in text for k in ("housing", "engine", "wheel", "motor", "gearbox", "cylinder", "block", "bearing")):
        return "housing"
    return "default"


# ---------------------------------------------------------------------------
# 4. Standard DXF Unit Resolution
# ---------------------------------------------------------------------------
UNIT_TO_MM = {
    1: 25.4,     # Inches
    2: 304.8,    # Feet
    4: 1.0,      # Millimeters
    5: 10.0,     # Centimeters
    6: 1000.0,   # Meters
}

def resolve_drawing_units(insunits: int) -> Dict[str, Any]:
    """Map DXF $INSUNITS header to mm scale factor."""
    if insunits in UNIT_TO_MM:
        return {
            "scale": UNIT_TO_MM[insunits],
            "units": "in" if insunits == 1 else "mm",
            "source": "header",
            "confidence": 0.99
        }
    return {
        "scale": 1.0,
        "units": "mm",
        "source": "assumed_mm",
        "confidence": 0.30
    }
