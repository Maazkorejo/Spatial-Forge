"""
SpatialForge Template Registry & Mechanical Topology Classifier.

Provides canonical relational Assemblies with bounded deformation ranges for
common mechanical classes, preventing topological hallucinations by construction.
"""
from typing import Dict, Any, Tuple, Optional
import json
import re
import cv2
import numpy as np

from schema import Assembly


def _d(v: float, tol: float = 0.35, src: str = "inferred") -> Dict[str, Any]:
    """Helper to define a dimension with bounded ranges."""
    return {
        "value": round(float(v), 2),
        "min": round(float(v * (1.0 - tol)), 2),
        "max": round(float(v * (1.0 + tol)), 2),
        "source": src
    }


# ---------------------------------------------------------------------------
# 1. TEMPLATE: Stacked Housing (Compact Cubic Machine / Gearbox / Mechanism)
# ---------------------------------------------------------------------------
def stacked_housing() -> Assembly:
    """
    Compact, stacked, roughly cubic machine:
    - Base mounting plate
    - Central cubic/chamfered main housing
    - Top cylindrical gear / ribbed cap
    - Large front-facing dial / lens
    - Side-mounted drive gear / handwheel
    """
    return Assembly.model_validate({
        "observations": [
            "Compact, stacked mechanical housing with clustered controls and gears.",
            "Central cubic body with a large front dial and top-mounted cylindrical gear cap.",
            "Side-mounted gear/wheel flush against the right face.",
            "Mounted on a low rectangular base plate with mounting ears."
        ],
        "object_class": "stacked_housing",
        "view_hint": {"projection": "isometric", "azimuth_deg": -32, "elevation_deg": 28},
        "parts": [
            # 1. Base Mounting Plate
            {
                "id": "base_mount",
                "label": "mounting plate",
                "kind": "rounded_box",
                "fillet": 3,
                "width_x": _d(75.0, 0.3),
                "depth_y": _d(65.0, 0.3),
                "height_z": _d(8.0, 0.4)
            },
            # 2. Main Housing Block (Sits centrally on base plate)
            {
                "id": "main_housing",
                "label": "central housing",
                "kind": "rounded_box",
                "fillet": 4,
                "placement": {
                    "parent": "base_mount",
                    "face": "top",
                    "align_x": "center",
                    "align_y": "center"
                },
                "width_x": _d(62.0, 0.3),
                "depth_y": _d(54.0, 0.3),
                "height_z": _d(52.0, 0.35)
            },
            # 3. Top Cylindrical Gear Cap (Ribbed top module)
            {
                "id": "top_cap",
                "label": "top gear cap",
                "kind": "cylinder",
                "axis": "z",
                "placement": {
                    "parent": "main_housing",
                    "face": "top",
                    "align_x": "center",
                    "align_y": "center",
                    "embed": 0.05
                },
                "diameter": _d(28.0, 0.35),
                "length": _d(16.0, 0.4)
            },
            # 4. Large Front Dial / Optical Hub (Mounted flush to front face)
            {
                "id": "front_dial",
                "label": "large front dial",
                "kind": "cylinder",
                "axis": "y",
                "placement": {
                    "parent": "main_housing",
                    "face": "front",
                    "align_x": "center",
                    "align_z": "center",
                    "embed": 0.05
                },
                "diameter": _d(32.0, 0.35),
                "length": _d(8.0, 0.4)
            },
            # 5. Side Drive Wheel / Gear (Mounted to right face)
            {
                "id": "side_wheel",
                "label": "side drive wheel",
                "kind": "cylinder",
                "axis": "x",
                "placement": {
                    "parent": "main_housing",
                    "face": "right",
                    "align_y": "center",
                    "align_z": "center",
                    "embed": 0.05
                },
                "diameter": _d(34.0, 0.35),
                "length": _d(7.0, 0.4)
            }
        ],
        "features": [
            {
                "id": "dial_knob",
                "kind": "dial",
                "layer": "feature",
                "on": {"part": "main_housing", "face": "front", "u": 0.78, "v": 0.75},
                "diameter": _d(9.0, 0.3),
                "depth": _d(4.0, 0.5, "assumed")
            },
            {
                "id": "switch_1",
                "kind": "button",
                "layer": "feature",
                "on": {"part": "main_housing", "face": "front", "u": 0.22, "v": 0.75},
                "diameter": _d(4.5, 0.3),
                "depth": _d(2.0, 0.5, "assumed")
            },
            {
                "id": "screw_fl",
                "kind": "screw",
                "layer": "cosmetic",
                "on": {"part": "base_mount", "face": "top", "u": 0.12, "v": 0.12},
                "diameter": _d(3.2, 0.2),
                "depth": _d(1.0, 0.5, "assumed")
            },
            {
                "id": "screw_fr",
                "kind": "screw",
                "layer": "cosmetic",
                "on": {"part": "base_mount", "face": "top", "u": 0.88, "v": 0.12},
                "diameter": _d(3.2, 0.2),
                "depth": _d(1.0, 0.5, "assumed")
            }
        ],
        "confidence": 0.88
    })


# ---------------------------------------------------------------------------
# 2. TEMPLATE: Sewing Machine (Cantilevered Arm + Bed + Tower)
# ---------------------------------------------------------------------------
def sewing_machine() -> Assembly:
    """Canonical sewing machine: base bed + tower + cantilevered arm + needle bar + handwheel."""
    from examples import sewing_machine as sm_canonical
    return sm_canonical()


# ---------------------------------------------------------------------------
# 3. TEMPLATE: Bench Grinder (Pedestal + Twin Symmetrical Grinding Wheels)
# ---------------------------------------------------------------------------
def bench_grinder() -> Assembly:
    """Bench grinder: pedestal base + motor cylinder + left and right grinding wheels."""
    return Assembly.model_validate({
        "observations": [
            "Bench grinder with heavy central pedestal, horizontal motor, and twin abrasive wheels."
        ],
        "object_class": "bench_grinder",
        "view_hint": {"projection": "isometric", "azimuth_deg": -30, "elevation_deg": 22},
        "parts": [
            {
                "id": "pedestal_base",
                "label": "pedestal base",
                "kind": "rounded_box",
                "fillet": 4,
                "width_x": _d(55.0, 0.3),
                "depth_y": _d(50.0, 0.3),
                "height_z": _d(14.0, 0.3)
            },
            {
                "id": "motor_housing",
                "label": "motor cylinder",
                "kind": "cylinder",
                "axis": "x",
                "placement": {
                    "parent": "pedestal_base",
                    "face": "top",
                    "align_x": "center",
                    "align_y": "center"
                },
                "diameter": _d(36.0, 0.3),
                "length": _d(52.0, 0.3)
            },
            {
                "id": "wheel_left",
                "label": "left grinding wheel",
                "kind": "cylinder",
                "axis": "x",
                "placement": {
                    "parent": "motor_housing",
                    "face": "left",
                    "align_y": "center",
                    "align_z": "center"
                },
                "diameter": _d(48.0, 0.25),
                "length": _d(12.0, 0.3)
            },
            {
                "id": "wheel_right",
                "label": "right grinding wheel",
                "kind": "cylinder",
                "axis": "x",
                "placement": {
                    "parent": "motor_housing",
                    "face": "right",
                    "align_y": "center",
                    "align_z": "center"
                },
                "diameter": _d(48.0, 0.25),
                "length": _d(12.0, 0.3)
            }
        ],
        "features": [],
        "confidence": 0.85
    })


# ---------------------------------------------------------------------------
# 4. TEMPLATE: Bracket / Mounting Flange (Structural L-Plate)
# ---------------------------------------------------------------------------
def bracket_flange() -> Assembly:
    """L-bracket / mounting plate with reinforced web."""
    return Assembly.model_validate({
        "observations": [
            "Right-angle mounting bracket with base flange, upright flange, and stiffener."
        ],
        "object_class": "bracket_flange",
        "view_hint": {"projection": "isometric", "azimuth_deg": -40, "elevation_deg": 30},
        "parts": [
            {
                "id": "base_flange",
                "label": "horizontal flange",
                "kind": "rounded_box",
                "fillet": 2,
                "width_x": _d(80.0, 0.3),
                "depth_y": _d(50.0, 0.3),
                "height_z": _d(8.0, 0.3)
            },
            {
                "id": "upright_flange",
                "label": "vertical flange",
                "kind": "rounded_box",
                "fillet": 2,
                "placement": {
                    "parent": "base_flange",
                    "face": "top",
                    "align_x": "min",
                    "align_y": "center"
                },
                "width_x": _d(10.0, 0.3),
                "depth_y": _d(50.0, 0.3),
                "height_z": _d(65.0, 0.3)
            }
        ],
        "features": [],
        "confidence": 0.85
    })


# ---------------------------------------------------------------------------
# 5. TEMPLATE: Centrifugal Pump (Volute Casing + Impeller + Flanges)
# ---------------------------------------------------------------------------
def centrifugal_pump() -> Assembly:
    """Industrial centrifugal pump: base skid, volute casing, suction and discharge flanges."""
    return Assembly.model_validate({
        "observations": [
            "Centrifugal pump with spiral volute body, horizontal suction port, vertical discharge port."
        ],
        "object_class": "centrifugal_pump",
        "view_hint": {"projection": "isometric", "azimuth_deg": -35, "elevation_deg": 25},
        "parts": [
            {
                "id": "skid_base",
                "label": "base plate",
                "kind": "rounded_box",
                "fillet": 3,
                "width_x": _d(90.0, 0.25),
                "depth_y": _d(60.0, 0.25),
                "height_z": _d(10.0, 0.3)
            },
            {
                "id": "volute_body",
                "label": "spiral volute casing",
                "kind": "cylinder",
                "axis": "y",
                "placement": {
                    "parent": "skid_base",
                    "face": "top",
                    "align_x": "center",
                    "align_y": "center"
                },
                "diameter": _d(55.0, 0.3),
                "length": _d(32.0, 0.3)
            },
            {
                "id": "discharge_neck",
                "label": "vertical discharge port",
                "kind": "cylinder",
                "axis": "z",
                "placement": {
                    "parent": "volute_body",
                    "face": "top",
                    "align_x": "center",
                    "align_y": "center"
                },
                "diameter": _d(22.0, 0.3),
                "length": _d(24.0, 0.35)
            },
            {
                "id": "suction_flange",
                "label": "axial suction flange",
                "kind": "cylinder",
                "axis": "y",
                "placement": {
                    "parent": "volute_body",
                    "face": "front",
                    "align_x": "center",
                    "align_z": "center"
                },
                "diameter": _d(30.0, 0.25),
                "length": _d(8.0, 0.3)
            }
        ],
        "features": [
            {
                "id": "drain_plug",
                "kind": "screw",
                "layer": "cosmetic",
                "on": {"part": "volute_body", "face": "bottom", "u": 0.5, "v": 0.5},
                "diameter": _d(4.0, 0.3),
                "depth": _d(2.0, 0.5, "assumed")
            }
        ],
        "confidence": 0.86
    })


# Registry Mapping
TEMPLATE_REGISTRY = {
    "stacked_housing": stacked_housing,
    "sewing_machine": sewing_machine,
    "bench_grinder": bench_grinder,
    "bracket_flange": bracket_flange,
    "centrifugal_pump": centrifugal_pump
}


# ---------------------------------------------------------------------------
# 5. Vision / Geometric Topology Classifier
# ---------------------------------------------------------------------------
def classify_sketch(img_bgr: np.ndarray, mask: np.ndarray, vlm_client: Any = None) -> Tuple[str, float, str]:
    """
    Classifies an input sketch into a mechanical topology class.
    Uses VLM if available, with robust geometric / aspect-ratio / contour fallback.

    Returns:
        (class_name, confidence, reason)
    """
    # 1. Check with VLM if client provided
    if vlm_client is not None:
        try:
            ok, png_buf = cv2.imencode(".png", img_bgr)
            if ok:
                png_bytes = png_buf.tobytes()
                system = "You are an expert mechanical CAD engineer classifying engineering sketches and blueprints."
                prompt = (
                    "Identify the mechanical topology class of this engineering sketch.\n"
                    "Options:\n"
                    "- 'stacked_housing': compact cubic or stacked housing with dials, gears, or wheels clustered on one body\n"
                    "- 'sewing_machine': elongated base bed with a vertical tower and a long horizontal cantilevered arm\n"
                    "- 'bench_grinder': central motor cylinder with twin circular grinding wheels on left and right\n"
                    "- 'bracket_flange': right-angle L-bracket or flat mounting plate\n"
                    "- 'centrifugal_pump': spiral or volute casing with inlet/outlet pipes\n"
                    "Return ONLY a JSON: {\"class\": \"...\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}"
                )
                resp = vlm_client.complete(system, prompt, images=[png_bytes])
                m = re.search(r"\{.*\}", resp, re.DOTALL)
                raw_json = m.group(0) if m else resp
                data = json.loads(raw_json)
                cand_cls = str(data.get("class", "")).strip().lower().replace(" ", "_")
                confidence = float(data.get("confidence", 0.9))
                reason = str(data.get("reason", f"VLM classified as {cand_cls}"))
                if cand_cls in TEMPLATE_REGISTRY:
                    return cand_cls, confidence, reason
        except Exception:
            pass

    # 2. Geometric Contour & Bounding Aspect Ratio Fallback
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return "stacked_housing", 0.5, "Empty mask fallback"

    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    aspect = w / max(h, 1.0)  # width / height

    # Contour convexity analysis
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.8
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area

    # Elongated with large open C-space (low solidity) -> Sewing Machine
    # e.g., Sewing machines have an aspect ratio > 1.4 and a large gap under the arm (solidity < 0.65)
    if aspect > 1.45 and solidity < 0.68:
        return "sewing_machine", 0.85, f"Elongated aspect ({aspect:.2f}) with open cantilever bay (solidity {solidity:.2f})"

    # Compact, stacked, roughly square/cubic (aspect between 0.85 and 1.35, high solidity) -> Stacked Housing
    if 0.75 <= aspect <= 1.35 and solidity >= 0.70:
        return "stacked_housing", 0.92, f"Compact stacked cubic silhouette (aspect {aspect:.2f}, solidity {solidity:.2f})"

    # Twin wide symmetry -> Bench Grinder
    if aspect > 1.5 and solidity >= 0.75:
        return "bench_grinder", 0.80, f"Wide horizontal layout with high solidity (aspect {aspect:.2f})"

    # Default to stacked_housing for compact devices
    return "stacked_housing", 0.82, f"Clustered mechanical device (aspect {aspect:.2f})"
