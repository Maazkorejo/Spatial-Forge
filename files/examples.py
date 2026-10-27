"""Worked example in the DSL. Used as a few-shot in the prompt and as ground truth in demo.py."""
from schema import Assembly


def _d(v, tol=0.3, src="inferred"):
    return {"value": v, "min": round(v * (1 - tol), 2), "max": round(v * (1 + tol), 2), "source": src}


def sewing_machine() -> Assembly:
    return Assembly.model_validate({
        "observations": [
            "Hand-drawn sewing machine seen from the front-left, slightly above (isometric-like).",
            "Long flat base bed; a vertical tower at the right end; an arm bridges leftwards over the bed.",
            "A thin needle bar hangs from the left end of the arm; a handwheel sits on the right end of the tower.",
            "Construction lines and shading ignored.",
        ],
        "object_class": "sewing machine",
        "view_hint": {"projection": "isometric", "azimuth_deg": -35, "elevation_deg": 25},
        "parts": [
            {"id": "base", "label": "base bed", "kind": "box", "fillet": 3,
             "width_x": _d(100), "depth_y": _d(44), "height_z": _d(9)},
            {"id": "tower", "label": "vertical tower", "kind": "box", "fillet": 4,
             "placement": {"parent": "base", "face": "top", "align_x": "max", "align_y": "center"},
             "width_x": _d(22), "depth_y": _d(28), "height_z": _d(48)},
            {"id": "arm", "label": "overhanging arm", "kind": "box", "fillet": 4,
             "placement": {"parent": "tower", "face": "top", "align_x": "max", "align_y": "center"},
             "width_x": _d(82), "depth_y": _d(28), "height_z": _d(17)},
            {"id": "needle", "label": "needle bar", "kind": "cylinder", "axis": "z",
             "placement": {"parent": "arm", "face": "bottom", "align_x": "min", "offset_x": 6,
                           "align_y": "center", "embed": 0.1},
             "diameter": _d(5, 0.4), "length": _d(26, 0.4)},
            {"id": "wheel", "label": "handwheel", "kind": "cylinder", "axis": "x",
             "placement": {"parent": "tower", "face": "right", "align_y": "center",
                           "align_z": "center", "offset_z": 8},
             "diameter": _d(20), "length": _d(5, 0.4)},
            {"id": "plate_slot", "label": "needle plate cutout", "kind": "box", "op": "difference",
             "cuts": ["base"],
             "placement": {"parent": "base", "face": "top", "align_x": "min", "offset_x": 18.5,
                           "align_y": "center", "embed": 0.8},
             "width_x": _d(16, 0.4, "assumed"), "depth_y": _d(12, 0.4, "assumed"),
             "height_z": _d(7, 0.4, "assumed")},
        ],
        "confidence": 0.7,
    })
