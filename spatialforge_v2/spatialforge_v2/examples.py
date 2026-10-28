"""Worked example in the DSL (v2): rounded base, tower, extruded sloped arm, handwheel, features.
Used as the few-shot in the prompt and as ground truth in the tests."""
from schema import Assembly


def _d(v, tol=0.3, src="inferred"):
    return {"value": v, "min": round(v * (1 - tol), 2), "max": round(v * (1 + tol), 2), "source": src}


def sewing_machine() -> Assembly:
    return Assembly.model_validate({
        "observations": [
            "Hand-drawn sewing machine seen from the front-left, slightly above (isometric-like).",
            "Long rounded base bed; a vertical tower at the right end; an arm with a sloped front end bridges left.",
            "A thin needle bar hangs from the left end of the arm; a handwheel sits on the right end of the tower.",
            "Front of the arm has one dial and two small buttons; the bed has a needle-plate slot.",
            "Construction lines and shading ignored.",
        ],
        "object_class": "sewing machine",
        "view_hint": {"projection": "isometric", "azimuth_deg": -35, "elevation_deg": 25},
        "parts": [
            {"id": "base", "label": "base bed", "kind": "rounded_box", "fillet": 3,
             "width_x": _d(100), "depth_y": _d(44), "height_z": _d(9)},
            {"id": "tower", "label": "vertical tower", "kind": "rounded_box", "fillet": 3,
             "placement": {"parent": "base", "face": "top", "align_x": "max", "align_y": "center"},
             "width_x": _d(22), "depth_y": _d(28), "height_z": _d(48)},
            {"id": "arm", "label": "arm with sloped nose", "kind": "extrusion",
             "profile": [[0.04, 0.0], [1.0, 0.0], [1.0, 1.0], [0.12, 1.0], [0.0, 0.55]],
             "placement": {"parent": "tower", "face": "top", "align_x": "max", "align_y": "center"},
             "width_x": _d(82), "depth_y": _d(28), "height_z": _d(17)},
            {"id": "needle", "label": "needle bar", "kind": "cylinder", "axis": "z",
             "placement": {"parent": "arm", "face": "bottom", "align_x": "min", "offset_x": 8,
                           "align_y": "center", "embed": 0.1},
             "diameter": _d(5, 0.4), "length": _d(26, 0.4)},
            {"id": "wheel", "label": "handwheel", "kind": "cylinder", "axis": "x",
             "placement": {"parent": "tower", "face": "right", "align_y": "center",
                           "align_z": "center", "offset_z": 8},
             "diameter": _d(20), "length": _d(5, 0.4)},
        ],
        "features": [
            {"id": "dial_1", "kind": "dial", "on": {"part": "arm", "face": "front", "u": 0.55, "v": 0.5},
             "diameter": _d(9, 0.3), "depth": _d(3, 0.5, "assumed")},
            {"id": "button_1", "kind": "button", "on": {"part": "arm", "face": "front", "u": 0.72, "v": 0.62},
             "diameter": _d(3.5, 0.3), "depth": _d(1.2, 0.5, "assumed")},
            {"id": "button_2", "kind": "button", "on": {"part": "arm", "face": "front", "u": 0.8, "v": 0.62},
             "diameter": _d(3.5, 0.3), "depth": _d(1.2, 0.5, "assumed")},
            {"id": "slot_1", "kind": "slot", "on": {"part": "base", "face": "top", "u": 0.3, "v": 0.5},
             "size_u": _d(16, 0.3), "size_v": _d(10, 0.3), "depth": _d(5, 0.5, "assumed")},
            {"id": "screw_1", "kind": "screw", "layer": "cosmetic",
             "on": {"part": "tower", "face": "front", "u": 0.5, "v": 0.2},
             "diameter": _d(3, 0.3), "depth": _d(0.8, 0.5, "assumed")},
        ],
        "confidence": 0.7,
    })
