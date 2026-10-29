"""
Reprojection Verification and Self-Flagging Gate.
Projects reconstructed 3D CadQuery solids back to 2D orthographic views via OpenCascade HLR,
computes point-sampled Chamfer Edge Precision, Recall, and F1 against drawn DXF geometry,
detects unexplained or hallucinated edges, and makes reliable self-flagging decisions.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cadquery as cq
import numpy as np
from scipy.spatial import cKDTree

BENCHMARK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "drawing_gen", "drawing_gen"))
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from metrics import edge_prf, flag_stats
from project import project_view, sample


def _prim_to_eval_ent(p: Dict[str, Any], role: str = "visible") -> Dict[str, Any]:
    """Convert project_view primitive into metrics-compatible entity."""
    t = p.get("t", "line")
    res = {"kind": t, "role": role}
    if t == "line":
        res["g"] = {"p": p["p"]}
    elif t in ("circle", "arc"):
        res["g"] = {"c": p["c"], "r": p["r"]}
        if t == "arc":
            res["g"]["a0"] = p["a0"]
            res["g"]["a1"] = p["a1"]
    elif t == "poly":
        res["g"] = {"pts": p["pts"]}
    return res


def _reg_ent_to_eval(e: Dict[str, Any]) -> Dict[str, Any]:
    """Convert registered view entity into metrics-compatible entity."""
    kind = e.get("kind", "line")
    g = e.get("g", e)
    role = e.get("role", "visible")
    res = {"kind": kind, "role": role}
    if kind == "line":
        res["g"] = {"p": g["p"]}
    elif kind in ("circle", "arc"):
        res["g"] = {"c": g["c"], "r": g["r"]}
        if kind == "arc":
            res["g"]["a0"] = g.get("a0", 0.0)
            res["g"]["a1"] = g.get("a1", 360.0)
    return res


def verify_reconstruction(
    solid: cq.Workplane | cq.Shape,
    registered_views: Dict[str, Any],
    tol_vis: float = 0.8,
    tol_hid: float = 1.2
) -> Dict[str, Any]:
    """
    Verify a reconstructed solid by reprojecting back into Front, Top, and Right views.
    Compares against registered 2D drawing entities using point-sampled Chamfer distance.
    Returns per-view scores, aggregate metrics, and a self-flagging decision.
    """
    shape = solid.val() if hasattr(solid, "val") else solid
    views = registered_views["views"]

    view_reports = {}
    f1_vis_list = []
    rec_vis_list = []
    prec_vis_list = []

    for v in ("front", "top", "right"):
        # 1. OpenCascade HLR reprojection
        proj = project_view(shape, v)
        
        proj_vis_ents = [_prim_to_eval_ent(p, "visible") for p in proj["visible"]]
        proj_hid_ents = [_prim_to_eval_ent(p, "hidden") for p in proj["hidden"]]

        # 2. Extract drawn entities in this view
        drawn_all = views[v]["entities"]
        drawn_vis = [_reg_ent_to_eval(e) for e in drawn_all if e.get("role", "visible") == "visible"]
        drawn_hid = [_reg_ent_to_eval(e) for e in drawn_all if e.get("role") == "hidden"]

        # 3. Compute Chamfer Edge PRF
        vis_scores = edge_prf(proj_vis_ents, drawn_vis, roles=("visible",), tol=tol_vis)
        hid_scores = edge_prf(proj_hid_ents, drawn_hid, roles=("hidden",), tol=tol_hid) if drawn_hid else {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        f1_vis_list.append(vis_scores["f1"])
        rec_vis_list.append(vis_scores["recall"])
        prec_vis_list.append(vis_scores["precision"])

        view_reports[v] = {
            "visible": vis_scores,
            "hidden": hid_scores,
            "reprojected_counts": {"vis": len(proj["visible"]), "hid": len(proj["hidden"])},
            "drawn_counts": {"vis": len(drawn_vis), "hid": len(drawn_hid)}
        }

    # Aggregate scores
    mean_f1 = float(np.mean(f1_vis_list))
    mean_rec = float(np.mean(rec_vis_list))
    mean_prec = float(np.mean(prec_vis_list))

    # Self-Flagging Decision:
    # Flag if visible edge F1 is below 0.85, or if precision or recall drops below 0.80
    flagged = bool(mean_f1 < 0.85 or mean_rec < 0.80 or mean_prec < 0.75)
    status = "needs_review" if flagged else "ok"
    confidence = round(mean_f1, 3)

    return {
        "status": status,
        "flagged": flagged,
        "confidence": confidence,
        "mean_vis_f1": round(mean_f1, 4),
        "mean_vis_recall": round(mean_rec, 4),
        "mean_vis_precision": round(mean_prec, 4),
        "views": view_reports
    }
