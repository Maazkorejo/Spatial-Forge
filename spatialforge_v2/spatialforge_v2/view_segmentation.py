"""
View Segmentation, Registration, and Projection Standard Inference.
Clusters normalized 2D drawing entities into orthographic views (Front, Top, Right/Side),
identifies first-angle vs third-angle projection, and registers view coordinates to 3D model space.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box as shapely_box


def _ent_bbox(e: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return [min_x, min_y, max_x, max_y] for an entity."""
    kind = e.get("kind", "")
    g = e.get("g", e)
    if kind == "line" or "p" in g:
        x0, y0, x1, y1 = g["p"]
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    if kind in ("circle", "arc") or "c" in g:
        cx, cy = g["c"]
        r = g["r"]
        return cx - r, cy - r, cx + r, cy + r
    return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# 1. Spatial View Clustering
# ---------------------------------------------------------------------------
def cluster_drawing_entities(
    entities: List[Dict[str, Any]],
    cluster_tol: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Cluster drawing geometry into distinct view regions.
    Filters out non-view annotations (title blocks, isolated dimensions).
    Uses spatial proximity and bounding box overlaps.
    """
    # 1. Keep only structural geometry (visible, hidden, center marks on features)
    geom_ents = []
    for e in entities:
        role = e.get("role", "visible")
        if role not in ("title", "dim"):
            g = e.get("g", e)
            if "p" in g or "c" in g:
                geom_ents.append(e)
            
    if not geom_ents:
        return []

    # 2. Extract bounding boxes and centers
    bboxes = [_ent_bbox(e) for e in geom_ents]
    centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in bboxes])

    # 3. Disjoint Set Union (DSU) based on proximity
    n = len(geom_ents)
    parent = list(range(n))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Connect entities that are within cluster_tol or whose expanded bboxes overlap
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=cluster_tol)
    for i, j in pairs:
        union(i, j)

    # Secondary check: if bounding boxes are very close along both axes
    for i in range(n):
        bi = bboxes[i]
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            bj = bboxes[j]
            # Check gap between boxes
            gap_x = max(0.0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            gap_y = max(0.0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            if gap_x < cluster_tol and gap_y < cluster_tol:
                union(i, j)

    # 4. Form clusters
    raw_clusters = defaultdict(list)
    for i in range(n):
        raw_clusters[find(i)].append(i)

    clusters: List[Dict[str, Any]] = []
    for root, idxs in raw_clusters.items():
        # Discard tiny noise clusters (e.g., isolated single short lines)
        if len(idxs) < 2:
            continue
        c_ents = [geom_ents[idx] for idx in idxs]
        c_bboxes = [bboxes[idx] for idx in idxs]
        min_x = min(b[0] for b in c_bboxes)
        min_y = min(b[1] for b in c_bboxes)
        max_x = max(b[2] for b in c_bboxes)
        max_y = max(b[3] for b in c_bboxes)
        w = max_x - min_x
        h = max_y - min_y
        
        # Must have significant area to be an orthographic view
        if w > 5.0 and h > 5.0:
            clusters.append({
                "bbox": [round(min_x, 3), round(min_y, 3), round(max_x, 3), round(max_y, 3)],
                "center": [round((min_x + max_x) / 2.0, 3), round((min_y + max_y) / 2.0, 3)],
                "width": round(w, 3),
                "height": round(h, 3),
                "entities": c_ents
            })

    # Sort clusters by area descending
    clusters.sort(key=lambda c: c["width"] * c["height"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# 2. View Role Identification & Standard Inference
# ---------------------------------------------------------------------------
def identify_views(
    clusters: List[Dict[str, Any]],
    texts: Optional[List[Dict[str, Any]]] = None,
    align_tol: float = 3.0
) -> Dict[str, Any]:
    """
    Identify front, top, and right views among the clusters using orthographic alignment.
    Determines first-angle vs third-angle projection.
    Returns:
      {
        "front": cluster,
        "top": cluster,
        "right": cluster,
        "standard": "third" | "first",
        "confidence": float
      }
    """
    if len(clusters) < 3:
        raise ValueError(f"Need at least 3 view clusters for orthographic reconstruction, found {len(clusters)}")

    # We take the top 3 largest clusters as the 3 primary orthographic views
    candidates = clusters[:3]

    # Look for text hint in title block
    standard_hint = None
    if texts:
        for t in texts:
            txt = t.get("text", "").upper()
            if "THIRD ANGLE" in txt:
                standard_hint = "third"
                break
            elif "FIRST ANGLE" in txt:
                standard_hint = "first"
                break

    # Determine which cluster is FRONT (the pivot of the L-shape)
    # Front must share X-span with Top and Y-span with Right
    best_pivot = None
    best_top = None
    best_right = None
    min_align_err = float("inf")

    for i in range(3):
        pivot = candidates[i]
        others = [candidates[j] for j in range(3) if j != i]
        c1, c2 = others[0], others[1]

        # Check alignment possibility A: c1 is top (X-aligned), c2 is right (Y-aligned)
        err_a_x = abs(pivot["bbox"][0] - c1["bbox"][0]) + abs(pivot["width"] - c1["width"])
        err_a_y = abs(pivot["bbox"][1] - c2["bbox"][1]) + abs(pivot["height"] - c2["height"])
        total_a = err_a_x + err_a_y

        # Check alignment possibility B: c2 is top (X-aligned), c1 is right (Y-aligned)
        err_b_x = abs(pivot["bbox"][0] - c2["bbox"][0]) + abs(pivot["width"] - c2["width"])
        err_b_y = abs(pivot["bbox"][1] - c1["bbox"][1]) + abs(pivot["height"] - c1["height"])
        total_b = err_b_x + err_b_y

        if total_a < min_align_err:
            min_align_err = total_a
            best_pivot = pivot
            best_top = c1
            best_right = c2
        if total_b < min_align_err:
            min_align_err = total_b
            best_pivot = pivot
            best_top = c2
            best_right = c1

    if best_pivot is None:
        raise ValueError("Could not find mutually aligned orthographic views.")

    # Determine Projection Standard from layout geometry
    # In Third Angle: Top view is ABOVE Front (y_top > y_front), Right view is to RIGHT (x_right > x_front)
    # In First Angle: Top view is BELOW Front (y_top < y_front), Right view is to LEFT (x_right < x_front)
    top_y = best_top["center"][1]
    front_y = best_pivot["center"][1]
    right_x = best_right["center"][0]
    front_x = best_pivot["center"][0]

    if top_y > front_y and right_x > front_x:
        inferred_standard = "third"
    elif top_y < front_y and right_x < front_x:
        inferred_standard = "first"
    else:
        # Fall back to text hint or default to third
        inferred_standard = standard_hint or "third"

    confidence = 1.0 if (standard_hint == inferred_standard) else 0.95

    return {
        "front": best_pivot,
        "top": best_top,
        "right": best_right,
        "standard": inferred_standard,
        "confidence": confidence,
        "alignment_error": round(min_align_err, 3)
    }


# ---------------------------------------------------------------------------
# 3. View Registration & Model Space Normalization
# ---------------------------------------------------------------------------
def register_views(
    identified: Dict[str, Any],
    scale: float = 1.0
) -> Dict[str, Any]:
    """
    Registers the 3 views into model space (mm):
      - Front view: (x, z)
      - Top view:   (x, y)
      - Right view: (y, z)
    Returns normalized geometry, view origins, and 3D bounding dimensions [W, D, H].
    """
    standard = identified["standard"]
    front = identified["front"]
    top = identified["top"]
    right = identified["right"]

    f_bx = front["bbox"]
    t_bx = top["bbox"]
    r_bx = right["bbox"]

    # View origins in paper coordinates
    u0_front, v0_front = f_bx[0], f_bx[1]
    u0_top, v0_top = t_bx[0], t_bx[1]
    u0_right, v0_right = r_bx[0], r_bx[1]

    # In model space:
    # W = Front width / scale
    # H = Front height / scale
    # D = Top height / scale (or Right width / scale)
    W = round(front["width"] / scale, 3)
    H = round(front["height"] / scale, 3)
    D = round(top["height"] / scale, 3)

    def normalize_entity(e: Dict[str, Any], u0: float, v0: float) -> Dict[str, Any]:
        """Convert entity from paper coords to local view coords in model mm."""
        kind = e.get("kind", "")
        g = e.get("g", e)
        res = dict(e)
        if kind == "line" or "p" in g or "p" in e:
            p_raw = g["p"] if "p" in g else e["p"]
            x0, y0, x1, y1 = p_raw
            p_norm = [
                round((x0 - u0) / scale, 4),
                round((y0 - v0) / scale, 4),
                round((x1 - u0) / scale, 4),
                round((y1 - v0) / scale, 4),
            ]
            res["g"] = {"p": p_norm}
            res["p"] = p_norm
            res["kind"] = "line"
        elif kind in ("circle", "arc") or "c" in g or "c" in e:
            c_raw = g["c"] if "c" in g else e["c"]
            r_raw = g["r"] if "r" in g else e["r"]
            cx, cy = c_raw
            r = r_raw
            c_norm = [round((cx - u0) / scale, 4), round((cy - v0) / scale, 4)]
            r_norm = round(r / scale, 4)
            res["g"] = {"c": c_norm, "r": r_norm}
            res["c"] = c_norm
            res["r"] = r_norm
            if kind == "arc" or "a0" in g or "a0" in e:
                res["g"]["a0"] = g.get("a0", e.get("a0", 0.0))
                res["g"]["a1"] = g.get("a1", e.get("a1", 360.0))
                res["a0"] = res["g"]["a0"]
                res["a1"] = res["g"]["a1"]
                res["kind"] = "arc"
            else:
                res["kind"] = "circle"
        return res

    norm_front = [normalize_entity(e, u0_front, v0_front) for e in front["entities"]]
    norm_top = [normalize_entity(e, u0_top, v0_top) for e in top["entities"]]
    norm_right = [normalize_entity(e, u0_right, v0_right) for e in right["entities"]]

    return {
        "standard": standard,
        "scale": scale,
        "dimensions": {"W": W, "D": D, "H": H},
        "origins": {
            "front": [round(u0_front, 3), round(v0_front, 3)],
            "top": [round(u0_top, 3), round(v0_top, 3)],
            "right": [round(u0_right, 3), round(v0_right, 3)],
        },
        "views": {
            "front": {
                "axes": ("x", "z"),
                "bbox_model": [0.0, 0.0, W, H],
                "entities": norm_front,
            },
            "top": {
                "axes": ("x", "y"),
                "bbox_model": [0.0, 0.0, W, D],
                "entities": norm_top,
            },
            "right": {
                "axes": ("y", "z"),
                "bbox_model": [0.0, 0.0, D, H],
                "entities": norm_right,
            },
        },
    }
