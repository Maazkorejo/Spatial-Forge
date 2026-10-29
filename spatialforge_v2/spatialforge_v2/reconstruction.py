"""
2.5D Feature-Based 3D CAD Reconstructor and Revolve Engine.
Synthesizes exact B-Rep solids from registered 2D orthographic views using CadQuery.
Reconstructs base extrusions, notches, top steps, through-holes, slots, counterbores,
blind pockets, and turned revolved shafts.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cadquery as cq
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize, unary_union

from dxf_ingest import build_closed_loops


def _cyl_z(d: float, h: float, x: float, y: float, z0: float = 0.0) -> cq.Workplane:
    """Create a vertical cylinder along Z."""
    return cq.Workplane("XY").circle(d / 2.0).extrude(h).translate((x, y, z0))


def _detect_family(registered: Dict[str, Any]) -> str:
    """Determine whether the part is turned (cylindrical revolution) or prismatic 2.5D."""
    views = registered["views"]
    r_ents = views["right"]["entities"]
    r_circles = [e for e in r_ents if e.get("kind") == "circle" or "c" in e.get("g", e)]
    r_lines = [e for e in r_ents if e.get("kind") == "line" or "p" in e.get("g", e)]

    # In a turned shaft, the side view is dominated by concentric circles
    if len(r_circles) >= 2 and len(r_lines) <= 4:
        return "turned"
    return "prismatic"


# ---------------------------------------------------------------------------
# 1. Prismatic 2.5D Feature Reconstruction
# ---------------------------------------------------------------------------
def reconstruct_prismatic(registered: Dict[str, Any]) -> Tuple[cq.Workplane, Dict[str, Any]]:
    """
    Reconstruct a prismatic 2.5D mechanical part feature-by-feature.
    """
    views = registered["views"]
    dims = registered["dimensions"]
    W, D, H = dims["W"], dims["D"], dims["H"]

    front_ents = views["front"]["entities"]
    top_ents = views["top"]["entities"]
    right_ents = views["right"]["entities"]

    features_detected = []

    # 1.1 Base Extrusion from Top View Footprint
    # Extract visible lines and arcs from top view
    top_vis_lines = [e for e in top_ents if (e.get("kind") == "line" or "p" in e.get("g", e)) and e.get("role", "visible") == "visible"]
    top_vis_arcs = [e for e in top_ents if (e.get("kind") == "arc" or ("c" in e.get("g", e) and "a0" in e.get("g", e))) and e.get("role", "visible") == "visible"]

    # View-local stitching closes remaining noise gaps on the view perimeter
    from dxf_ingest import stitch_endpoints
    st_top_lines, _ = stitch_endpoints(top_vis_lines, user_tolerance=1.5)

    polys, _ = build_closed_loops(st_top_lines, top_vis_arcs)

    nominal_area = W * D
    body = None

    if polys:
        # Find the outer footprint: union adjacent coplanar faces (e.g. step dividers)
        max_single = max(polys, key=lambda p: p.area)
        unioned = unary_union(polys)
        
        # If union forms a single outer polygon larger than individual partitions
        if hasattr(unioned, "geom_type") and unioned.geom_type == "Polygon":
            candidate = unioned
        elif hasattr(unioned, "geoms"):
            candidate = max(unioned.geoms, key=lambda p: p.area)
        else:
            candidate = max_single

        # Verify candidate polygon has substantial area (> 50% of W*D)
        if candidate.area >= 0.5 * nominal_area:
            ext_coords = list(candidate.exterior.coords)[:-1]
            try:
                body = cq.Workplane("XY").polyline(ext_coords).close().extrude(H)
                if nominal_area - candidate.area > 50.0:
                    features_detected.append({"type": "notch", "area_removed": round(nominal_area - candidate.area, 2)})
            except Exception:
                body = None

    if body is None:
        # Fallback to bounding box footprint
        body = cq.Workplane("XY").box(W, D, H, centered=False)

    # 1.2 Step Cut on Top (from Front View)
    front_vis_lines = [e for e in front_ents if (e.get("kind") == "line" or "p" in e.get("g", e)) and e.get("role", "visible") == "visible"]
    # Look for step in top-left or top-right: e.g. horizontal line at H - sz
    for l in front_vis_lines:
        g = l.get("g", l)
        x0, z0, x1, z1 = g["p"]
        if abs(z0 - z1) < 0.5:  # horizontal line
            z_step = (z0 + z1) / 2.0
            if 3.0 < z_step < H - 3.0:
                min_x, max_x = min(x0, x1), max(x0, x1)
                sx = max_x - min_x
                sz = H - z_step
                # Step at top-left
                if min_x < 2.0 and sx > 5.0:
                    body = body.cut(cq.Workplane("XY").box(sx, D + 2.0, sz, centered=False).translate((0.0, -1.0, z_step)))
                    features_detected.append({"type": "step", "sx": round(sx, 2), "sz": round(sz, 2)})
                    break
                # Step at top-right
                elif max_x > W - 2.0 and sx > 5.0:
                    body = body.cut(cq.Workplane("XY").box(sx, D + 2.0, sz, centered=False).translate((W - sx, -1.0, z_step)))
                    features_detected.append({"type": "step", "sx": round(sx, 2), "sz": round(sz, 2)})
                    break

    # 1.3 Vertical Holes & Counterbores (from Top View Circles)
    top_circles = [e for e in top_ents if e.get("kind") == "circle" or ("c" in e.get("g", e) and "a0" not in e.get("g", e))]
    
    # Group concentric circles
    circle_groups = []
    used_circles = set()
    for i, c1 in enumerate(top_circles):
        if i in used_circles:
            continue
        g1 = c1.get("g", c1)
        cx1, cy1, r1 = g1["c"][0], g1["c"][1], g1["r"]
        group = [(cx1, cy1, r1 * 2.0)]
        used_circles.add(i)
        for j, c2 in enumerate(top_circles):
            if j in used_circles:
                continue
            g2 = c2.get("g", c2)
            cx2, cy2, r2 = g2["c"][0], g2["c"][1], g2["r"]
            if math.hypot(cx2 - cx1, cy2 - cy1) < 1.0:
                group.append((cx2, cy2, r2 * 2.0))
                used_circles.add(j)
        circle_groups.append(group)

    # Hidden horizontal lines in front/right views (for counterbore/pocket depths)
    front_hidden_lines = [e for e in front_ents if (e.get("kind") == "line" or "p" in e.get("g", e)) and e.get("role") == "hidden"]
    front_hidden_z = []
    for l in front_hidden_lines:
        g = l.get("g", l)
        x0, z0, x1, z1 = g["p"]
        if abs(z0 - z1) < 0.5:
            front_hidden_z.append((min(x0, x1), max(x0, x1), (z0 + z1) / 2.0))

    for grp in circle_groups:
        if len(grp) == 1:
            # Simple through hole
            cx, cy, d = grp[0]
            body = body.cut(_cyl_z(d, H + 4.0, cx, cy, -2.0))
            features_detected.append({"type": "hole_z", "x": round(cx, 2), "y": round(cy, 2), "d": round(d, 2)})
        else:
            # Concentric: through hole + counterbore
            grp.sort(key=lambda c: c[2])  # sort by diameter
            cx, cy, d_hole = grp[0]
            _, _, d_cb = grp[1]
            # Cut through hole
            body = body.cut(_cyl_z(d_hole, H + 4.0, cx, cy, -2.0))
            # Find counterbore depth from hidden lines near cx
            cb_depth = round(H * 0.35, 2)  # default
            for min_x, max_x, z_floor in front_hidden_z:
                if min_x - 1.0 <= cx <= max_x + 1.0:
                    cb_depth = round(H - z_floor, 2)
                    break
            body = body.cut(_cyl_z(d_cb, cb_depth + 1.0, cx, cy, H - cb_depth))
            features_detected.append({"type": "hole_z", "x": round(cx, 2), "y": round(cy, 2), "d": round(d_hole, 2)})
            features_detected.append({"type": "counterbore", "x": round(cx, 2), "y": round(cy, 2), "d": round(d_cb, 2), "depth": cb_depth})

    # 1.4 Rectangular Pockets (Inner loops in Top View)
    if polys and len(polys) > 1:
        outer_poly = max(polys, key=lambda p: p.area)
        for poly in polys:
            if poly is outer_poly:
                continue
            min_x, min_y, max_x, max_y = poly.bounds
            sx = max_x - min_x
            sy = max_y - min_y
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            
            # Find pocket depth from hidden lines in Front view
            depth = round(H * 0.4, 2)
            is_bottom = False
            for hx0, hx1, z_floor in front_hidden_z:
                if abs(hx0 - min_x) < 2.0 or abs(hx1 - max_x) < 2.0 or (hx0 <= cx <= hx1):
                    if z_floor < H / 2.0:
                        # Could be pocket depth from bottom or deep pocket from top
                        depth = round(H - z_floor, 2)
                    else:
                        depth = round(H - z_floor, 2)
                    break

            body = body.cut(cq.Workplane("XY").box(sx, sy, depth + 1.0, centered=False).translate((min_x, min_y, H - depth)))
            features_detected.append({"type": "pocket_top", "x": round(min_x, 2), "y": round(min_y, 2), "sx": round(sx, 2), "sy": round(sy, 2), "depth": depth})

    return body, {
        "family": "prismatic",
        "dimensions": dims,
        "features": features_detected,
        "volume": float(body.val().Volume()),
        "valid": bool(body.val().isValid())
    }


# ---------------------------------------------------------------------------
# 2. Turned Shaft Feature Reconstruction (Revolve)
# ---------------------------------------------------------------------------
def reconstruct_turned(registered: Dict[str, Any]) -> Tuple[cq.Workplane, Dict[str, Any]]:
    """
    Reconstruct a turned cylindrical stepped shaft using CadQuery revolve.
    """
    views = registered["views"]
    dims = registered["dimensions"]
    L = dims["W"]

    front_ents = views["front"]["entities"]
    front_lines = [e for e in front_ents if e.get("kind") == "line" or "p" in e.get("g", e)]
    
    # 2.1 Revolve Profile Extraction from Front View (half-profile above centerline)
    # Filter lines above the axis (y >= -0.5)
    axis_y = dims["H"] / 2.0  # nominal center axis in registered view coords
    
    # Collect steps: horizontal segments at radius r = abs(y - axis_y)
    step_segments = []
    for l in front_lines:
        g = l.get("g", l)
        x0, y0, x1, y1 = g["p"]
        if abs(y0 - y1) < 0.5:  # horizontal segment
            x_min, x_max = min(x0, x1), max(x0, x1)
            if x_max - x_min > 4.0:
                r = abs(((y0 + y1) / 2.0) - axis_y)
                if r > 2.0:
                    step_segments.append((x_min, x_max, round(r, 2)))

    # Sort steps along X
    step_segments.sort(key=lambda s: s[0])
    
    # Construct half-profile vertices along X
    pts = [(0.0, 0.0)]
    ch = 1.5  # standard end chamfer
    
    if step_segments:
        r0 = step_segments[0][2]
        pts += [(0.0, max(0.0, r0 - ch)), (ch, r0)]
        for i, (x_min, x_max, r) in enumerate(step_segments):
            if i > 0:
                pts.append((x_min, r))
            if i < len(step_segments) - 1:
                pts.append((x_max, r))
            else:
                pts += [(x_max - ch, r), (x_max, max(0.0, r - ch)), (x_max, 0.0)]
    else:
        # Fallback to simple cylinder
        pts = [(0.0, 0.0), (0.0, dims["H"] / 2.0), (L, dims["H"] / 2.0), (L, 0.0)]

    # Revolve 360 degrees around X axis
    prof = cq.Workplane("XY").polyline(pts).close()
    body = prof.revolve(360, (0, 0, 0), (1, 0, 0))

    features_detected = []

    # 2.2 Cross-Hole (from Top View Circles along axis)
    top_ents = views["top"]["entities"]
    top_circles = [e for e in top_ents if e.get("kind") == "circle" or "c" in e.get("g", e)]
    for c in top_circles:
        g = c.get("g", c)
        cx, cy, r = g["c"][0], g["c"][1], g["r"]
        d = round(r * 2.0, 2)
        body = body.cut(_cyl_z(d, dims["H"] + 10.0, cx, 0.0, -dims["H"] / 2.0 - 5.0))
        features_detected.append({"type": "cross_hole", "x": round(cx, 2), "d": d})

    return body, {
        "family": "turned",
        "L": L,
        "features": features_detected,
        "volume": float(body.val().Volume()),
        "valid": bool(body.val().isValid())
    }


# ---------------------------------------------------------------------------
# 3. Main Reconstruction Dispatcher
# ---------------------------------------------------------------------------
def reconstruct(registered: Dict[str, Any]) -> Tuple[cq.Workplane, Dict[str, Any]]:
    """
    Main entrypoint: reconstructs a valid 3D B-Rep CAD model from registered 2D orthographic views.
    Dispatches to prismatic 2.5D or turned revolve reconstructor.
    """
    family = _detect_family(registered)
    if family == "turned":
        return reconstruct_turned(registered)
    else:
        return reconstruct_prismatic(registered)
