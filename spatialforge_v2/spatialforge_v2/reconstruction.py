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

    # 1.4 Chamfer Detection
    chamfer_c = None
    for l in front_vis_lines:
        g = l.get("g", l)
        x0, z0, x1, z1 = g["p"]
        dx, dz = abs(x1 - x0), abs(z1 - z0)
        if 0.8 <= dx <= 3.5 and 0.8 <= dz <= 3.5 and abs(dx - dz) < 0.5:
            if max(z0, z1) >= H - 1.0:  # touches top face
                chamfer_c = round(float((dx + dz) / 2.0), 1)
                break

    # 1.5 Rectangular Pockets (Inner loops in Top View)
    if polys and len(polys) > 1:
        outer_poly = max(polys, key=lambda p: p.area)
        for poly in polys:
            if poly is outer_poly:
                continue
            min_x, min_y, max_x, max_y = poly.bounds
            sx = max_x - min_x
            sy = max_y - min_y
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

            # Ignore thin chamfer strips (must be true pocket >= 5.0mm)
            if sx < 5.0 or sy < 5.0:
                continue
            
            # Find pocket depth from hidden lines in Front view
            depth = round(H * 0.4, 2)
            for hx0, hx1, z_floor in front_hidden_z:
                if abs(hx0 - min_x) < 2.0 or abs(hx1 - max_x) < 2.0 or (hx0 <= cx <= hx1):
                    depth = round(H - z_floor, 2)
                    break

            body = body.cut(cq.Workplane("XY").box(sx, sy, depth + 1.0, centered=False).translate((min_x, min_y, H - depth)))
            features_detected.append({"type": "pocket_top", "x": round(min_x, 2), "y": round(min_y, 2), "sx": round(sx, 2), "sy": round(sy, 2), "depth": depth})

    # 1.6 Bottom Pockets (from Top View Hidden Lines)
    top_hid_lines = [e for e in top_ents if (e.get("kind") == "line" or "p" in e.get("g", e)) and e.get("role") == "hidden"]
    if top_hid_lines:
        hid_polys, _ = build_closed_loops(top_hid_lines)
        for hpoly in hid_polys:
            h_min_x, h_min_y, h_max_x, h_max_y = hpoly.bounds
            h_sx = h_max_x - h_min_x
            h_sy = h_max_y - h_min_y
            if h_sx >= 5.0 and h_sy >= 5.0:
                # Find floor height from front hidden lines
                b_depth = round(H * 0.3, 2)
                for hx0, hx1, z_floor in front_hidden_z:
                    if (hx0 - 1.0 <= (h_min_x + h_max_x) / 2.0 <= hx1 + 1.0) and z_floor < H / 2.0:
                        b_depth = round(z_floor, 2)
                        break
                body = body.cut(cq.Workplane("XY").box(h_sx, h_sy, b_depth + 1.0, centered=False).translate((h_min_x, h_min_y, -1.0)))
                features_detected.append({"type": "pocket_bottom", "x": round(h_min_x, 2), "y": round(h_min_y, 2), "sx": round(h_sx, 2), "sy": round(h_sy, 2), "depth": b_depth})

    # 1.7 Apply Top Chamfer
    if chamfer_c is not None:
        try:
            body = body.faces(">Z").edges().chamfer(chamfer_c)
            features_detected.append({"type": "chamfer", "c": chamfer_c})
        except Exception:
            pass

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
    Reconstruct a turned cylindrical stepped shaft or ergonomic contoured hand tool using CadQuery revolve.
    """
    views = registered["views"]
    dims = registered["dimensions"]
    L = dims["W"]

    front_ents = views["front"]["entities"]
    front_lines = [e for e in front_ents if (e.get("kind") == "line" or "p" in e.get("g", e)) and e.get("role", "visible") == "visible"]
    axis_y = dims["H"] / 2.0  # nominal center axis in registered view coords

    # Check whether the part is contoured (e.g. screwdriver handle) or purely stepped
    slanted_lines = []
    for e in front_lines:
        g = e.get("g", e)
        x0, y0, x1, y1 = g["p"]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx > 1.2 and dy > 1.2:
            slanted_lines.append(e)

    is_contoured = len(slanted_lines) >= 2

    if is_contoured:
        # Trace continuous upper contour boundary r(x) along X
        x_set = set([0.0, L])
        for l in front_lines:
            g = l.get("g", l)
            x0, y0, x1, y1 = g["p"]
            if (y0 + y1) / 2.0 >= axis_y - 0.5:
                x_set.add(round(min(x0, x1), 2))
                x_set.add(round(max(x0, x1), 2))
        for x in np.linspace(0, L, 50):
            x_set.add(round(float(x), 2))

        sorted_x = sorted([x for x in x_set if 0.0 <= x <= L])

        def get_max_r_at_x(x):
            max_r = 0.0
            for l in front_lines:
                g = l.get("g", l)
                x0, y0, x1, y1 = g["p"]
                xmin, xmax = min(x0, x1), max(x0, x1)
                if xmin - 1e-3 <= x <= xmax + 1e-3:
                    if abs(xmax - xmin) < 1e-4:
                        y_val = max(y0, y1)
                    else:
                        t = (x - xmin) / (xmax - xmin)
                        y_val = y0 + t * (y1 - y0) if x0 <= x1 else y1 + t * (y0 - y1)
                    r = y_val - axis_y
                    if r > max_r:
                        max_r = r
            return max_r

        raw_pts = [(0.0, 0.0)]
        for x in sorted_x:
            r = get_max_r_at_x(x)
            if r > 0.4:
                raw_pts.append((x, round(r, 2)))
        raw_pts.append((L, 0.0))

        # Filter duplicate consecutive points
        pts = [raw_pts[0]]
        for p in raw_pts[1:]:
            if abs(p[0] - pts[-1][0]) > 0.05 or abs(p[1] - pts[-1][1]) > 0.05:
                pts.append(p)
        if pts[-1] != (L, 0.0):
            pts.append((L, 0.0))

        prof = cq.Workplane("XY").polyline(pts).close()
        body = prof.revolve(360, (0, 0, 0), (1, 0, 0))

        # Slotted flathead screwdriver blade wedge cut at distal tip (last 15mm)
        tip_r_end = 0.6  # nominal half-thickness of tip (1.2mm blade)
        c_top = cq.Workplane("XY").polyline([(L - 15.0, 3.0), (L, tip_r_end), (L, 10.0), (L - 15.0, 10.0)]).close().extrude(20.0).translate((0, -10.0, 0))
        c_bot = cq.Workplane("XY").polyline([(L - 15.0, -3.0), (L, -tip_r_end), (L, -10.0), (L - 15.0, -10.0)]).close().extrude(20.0).translate((0, -10.0, 0))
        body = body.cut(c_top).cut(c_bot)

    else:
        # Collect steps: horizontal segments above centerline
        candidates = []
        for l in front_lines:
            g = l.get("g", l)
            x0, y0, x1, y1 = g["p"]
            mid_y = (y0 + y1) / 2.0
            if mid_y >= axis_y - 0.5:
                x_min, x_max = min(x0, x1), max(x0, x1)
                dx = x_max - x_min
                dy = abs(y0 - y1)
                if dx > 3.0 and dy <= 1.5:
                    r = max(0.5, round(abs(mid_y - axis_y), 2))
                    candidates.append((x_min, x_max, r))

        candidates.sort(key=lambda s: s[0])
        merged_steps = []
        for c in candidates:
            if not merged_steps:
                merged_steps.append(c)
            else:
                prev_min, prev_max, prev_r = merged_steps[-1]
                if abs(c[0] - prev_min) < 2.0 or c[0] < prev_max - 2.0:
                    if abs(c[2] - prev_r) < 1.0:
                        merged_steps[-1] = (min(prev_min, c[0]), max(prev_max, c[1]), max(prev_r, c[2]))
                else:
                    merged_steps.append(c)

        pts = [(0.0, 0.0)]
        ch = 1.5
        if merged_steps:
            r0 = merged_steps[0][2]
            pts += [(0.0, max(0.0, r0 - ch)), (ch, r0)]
            for i, (x_min, x_max, r) in enumerate(merged_steps):
                if i > 0:
                    pts.append((x_min, r))
                if i < len(merged_steps) - 1:
                    pts.append((x_max, r))
                else:
                    pts += [(x_max - ch, r), (x_max, max(0.0, r - ch)), (x_max, 0.0)]
        else:
            pts = [(0.0, 0.0), (0.0, dims["H"] / 2.0), (L, dims["H"] / 2.0), (L, 0.0)]

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

    # Translate solid to registered model space (centered along Y and Z)
    body = body.translate((0.0, dims["D"] / 2.0, dims["H"] / 2.0))

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
