"""
DXF Ingestion, Normalization, Classification, and Adaptive Endpoint Stitching.
Converts raw 2D CAD drawings (clean or noisy) into clean, topologically verified planar entities
and closed Shapely loops with exact dimensions and scale regression.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import ezdxf
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union


def _xy(v) -> List[float]:
    return [float(v.x), float(v.y)]


# ---------------------------------------------------------------------------
# 1. Raw DXF Ingestion & Entity Extraction
# ---------------------------------------------------------------------------
def ingest_dxf(path: str) -> Dict[str, Any]:
    """
    Ingest a DXF file and extract normalized geometric entities.
    Expands blocks/inserts, decomposes polylines into lines and arcs,
    and extracts text and native dimensions.
    """
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    insunits = doc.header.get("$INSUNITS", 0)  # 4 = mm, 1 = inches, 0 = undefined
    
    # Check default document layers
    doc_layers = {layer.dxf.name: layer.dxf.get("linetype", "CONTINUOUS") for layer in doc.layers}
    
    raw_lines: List[Dict[str, Any]] = []
    raw_circles: List[Dict[str, Any]] = []
    raw_arcs: List[Dict[str, Any]] = []
    raw_texts: List[Dict[str, Any]] = []
    raw_dims: List[Dict[str, Any]] = []
    
    def process_entity(e):
        t = e.dxftype()
        lay = e.dxf.get("layer", "0")
        lt = e.dxf.get("linetype", "BYLAYER")
        if lt == "BYLAYER":
            lt = doc_layers.get(lay, "CONTINUOUS")
        lt = str(lt).upper()
        
        if t == "LINE":
            p1, p2 = _xy(e.dxf.start), _xy(e.dxf.end)
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 1e-5:
                raw_lines.append({"p": p1 + p2, "layer": lay, "ltype": lt, "handle": e.dxf.handle})
        elif t == "CIRCLE":
            c, r = _xy(e.dxf.center), float(e.dxf.radius)
            if r > 1e-5:
                raw_circles.append({"c": c, "r": r, "layer": lay, "ltype": lt, "handle": e.dxf.handle})
        elif t == "ARC":
            c, r = _xy(e.dxf.center), float(e.dxf.radius)
            a0, a1 = float(e.dxf.start_angle), float(e.dxf.end_angle)
            if r > 1e-5:
                raw_arcs.append({"c": c, "r": r, "a0": a0, "a1": a1, "layer": lay, "ltype": lt, "handle": e.dxf.handle})
        elif t in ("LWPOLYLINE", "POLYLINE"):
            # Decompose polyline vertices into lines / arcs (with bulge support)
            pts = list(e.get_points(format="xyb") if t == "LWPOLYLINE" else [(v.dxf.location.x, v.dxf.location.y, 0.0) for v in e.vertices])
            is_closed = e.is_closed
            n = len(pts)
            for i in range(n - (0 if is_closed else 1)):
                p1, p2 = pts[i], pts[(i + 1) % n]
                x1, y1, b = p1[0], p1[1], p1[2] if len(p1) > 2 else 0.0
                x2, y2 = p2[0], p2[1]
                if abs(b) < 1e-6:
                    if math.hypot(x2 - x1, y2 - y1) > 1e-5:
                        raw_lines.append({"p": [x1, y1, x2, y2], "layer": lay, "ltype": lt, "handle": e.dxf.handle})
                else:
                    # Bulge = tan(included_angle / 4)
                    theta = 4.0 * math.atan(b)
                    chord = math.hypot(x2 - x1, y2 - y1)
                    if chord > 1e-5:
                        r = abs(chord / (2.0 * math.sin(theta / 2.0)))
                        # Midpoint and normal
                        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        dx, dy = (x2 - x1) / chord, (y2 - y1) / chord
                        h = r * math.cos(theta / 2.0)
                        # Center direction depends on sign of bulge
                        cx = mx - dy * h if b > 0 else mx + dy * h
                        cy = my + dx * h if b > 0 else my - dx * h
                        a0 = math.degrees(math.atan2(y1 - cy, x1 - cx)) % 360.0
                        a1 = math.degrees(math.atan2(y2 - cy, x2 - cx)) % 360.0
                        raw_arcs.append({"c": [cx, cy], "r": r, "a0": a0, "a1": a1, "layer": lay, "ltype": lt, "handle": e.dxf.handle})
        elif t in ("TEXT", "MTEXT"):
            txt = e.dxf.text if t == "TEXT" else e.text
            raw_texts.append({"text": str(txt).strip(), "at": _xy(e.dxf.insert), "layer": lay, "handle": e.dxf.handle})
        elif t == "DIMENSION":
            raw_dims.append({
                "measurement": float(e.get_measurement()),
                "text": str(e.dxf.get("text", "<>")).strip(),
                "layer": lay,
                "type": e.dimtype,
                "handle": e.dxf.handle
            })
        elif t == "INSERT":
            # Flatten / explode blocks recursively
            for child in e.explode():
                process_entity(child)

    for ent in msp:
        process_entity(ent)
        
    return {
        "insunits": insunits,
        "units": "mm" if insunits == 4 else ("in" if insunits == 1 else "unknown"),
        "lines": raw_lines,
        "circles": raw_circles,
        "arcs": raw_arcs,
        "texts": raw_texts,
        "dims": raw_dims,
    }


# ---------------------------------------------------------------------------
# 2. Deduplication & Collinear Fragment Merging
# ---------------------------------------------------------------------------
def merge_collinear_lines(lines: List[Dict[str, Any]], ang_tol_deg: float = 1.0, dist_tol: float = 0.25) -> List[Dict[str, Any]]:
    """
    Merge overlapping or collinear line fragments into single continuous segments.
    Preserves layer and linetype where consistent.
    """
    if not lines:
        return []
    
    # Represent each line as a directed segment
    segs = []
    for l in lines:
        x0, y0, x1, y1 = l["p"]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-4:
            continue
        # Standardize direction: positive dx or if vertical positive dy
        if (x1 < x0) or (abs(x1 - x0) < 1e-5 and y1 < y0):
            x0, y0, x1, y1 = x1, y1, x0, y0
        ang = math.atan2(y1 - y0, x1 - x0) % math.pi
        segs.append({
            "p": [x0, y0, x1, y1],
            "ang": ang,
            "len": length,
            "layer": l.get("layer", "0"),
            "ltype": l.get("ltype", "CONTINUOUS"),
            "role": l.get("role", "unknown")
        })
    
    # Group segments with similar angles
    ang_tol_rad = math.radians(ang_tol_deg)
    merged: List[Dict[str, Any]] = []
    used = set()
    
    for i, s1 in enumerate(segs):
        if i in used:
            continue
        group = [i]
        used.add(i)
        
        # Unit direction vector of primary segment
        ux = math.cos(s1["ang"])
        uy = math.sin(s1["ang"])
        # Normal vector
        nx = -uy
        ny = ux
        
        o_x, o_y = s1["p"][0], s1["p"][1]
        
        for j, s2 in enumerate(segs):
            if j in used:
                continue
            # Angular difference
            d_ang = abs(s1["ang"] - s2["ang"])
            if d_ang > ang_tol_rad and abs(d_ang - math.pi) > ang_tol_rad:
                continue
            
            # Perpendicular distance to s1's line
            p2_0_dist = abs((s2["p"][0] - o_x) * nx + (s2["p"][1] - o_y) * ny)
            p2_1_dist = abs((s2["p"][2] - o_x) * nx + (s2["p"][3] - o_y) * ny)
            if p2_0_dist > dist_tol or p2_1_dist > dist_tol:
                continue
            
            # Check overlap or gap along primary direction
            # Project endpoints onto primary line
            t1_a = 0.0
            t1_b = (s1["p"][2] - o_x) * ux + (s1["p"][3] - o_y) * uy
            t2_a = (s2["p"][0] - o_x) * ux + (s2["p"][1] - o_y) * uy
            t2_b = (s2["p"][2] - o_x) * ux + (s2["p"][3] - o_y) * uy
            if t2_b < t2_a:
                t2_a, t2_b = t2_b, t2_a
            
            # Overlap or close gap
            min_t = min(t1_a, t2_a)
            max_t = max(t1_b, t2_b)
            span = max_t - min_t
            sum_len = (t1_b - t1_a) + (t2_b - t2_a)
            gap = span - sum_len
            
            # If segments overlap (gap <= 0) or are separated by a small gap <= dist_tol
            if gap <= dist_tol * 2.0:
                group.append(j)
                used.add(j)
                # Expand s1 bounds
                t1_a = min_t
                t1_b = max_t
                s1["p"] = [o_x + ux * t1_a, o_y + uy * t1_a, o_x + ux * t1_b, o_y + uy * t1_b]
        
        merged.append({
            "p": s1["p"],
            "layer": s1["layer"],
            "ltype": s1["ltype"],
            "role": s1["role"]
        })
        
    return merged


# ---------------------------------------------------------------------------
# 3. Line Classification & Dash/Explosion Recovery
# ---------------------------------------------------------------------------
def classify_and_filter_entities(ingested: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify raw DXF entities into semantic roles:
      - 'visible' (continuous contour lines, visible circles/arcs)
      - 'hidden'  (dashed hidden edges or reconstructed exploded dash chains)
      - 'center'  (centerlines)
      - 'dim'     (witness lines, dimension lines, arrows, numbers)
      - 'title'   (drawing border, title block)
    Handles Layer-0 collapse and exploded dashes.
    """
    lines = list(ingested["lines"])
    circles = list(ingested["circles"])
    arcs = list(ingested["arcs"])
    texts = list(ingested["texts"])
    dims = list(ingested["dims"])
    
    # 3.1 Initial classification from explicit layers/linetypes
    for l in lines:
        lay = l.get("layer", "").upper()
        lt = l.get("ltype", "").upper()
        if "HIDDEN" in lay or "DASH" in lt:
            l["role"] = "hidden"
        elif "CENTER" in lay or "CENTER" in lt:
            l["role"] = "center"
        elif "DIM" in lay:
            l["role"] = "dim"
        elif "TITLE" in lay:
            l["role"] = "title"
        elif lay == "VISIBLE":
            l["role"] = "visible"
        else:
            l["role"] = "unknown"
            
    # 3.2 Exploded Dash Recovery:
    # If dashes were exploded into continuous segments on layer 0, find collinear chains
    # of short segments (e.g., len ~ 1.5 - 3.5mm) separated by dash gaps (len ~ 0.5 - 2.0mm).
    short_segs = []
    other_lines = []
    for l in lines:
        if l["role"] == "unknown":
            x0, y0, x1, y1 = l["p"]
            length = math.hypot(x1 - x0, y1 - y0)
            if 0.8 <= length <= 4.0:
                short_segs.append(l)
            else:
                other_lines.append(l)
        else:
            other_lines.append(l)
            
    # Chain collinear short segments
    if len(short_segs) >= 2:
        merged_dashes = merge_collinear_lines(short_segs, ang_tol_deg=1.0, dist_tol=1.5)
        reconstructed_hidden = []
        for md in merged_dashes:
            x0, y0, x1, y1 = md["p"]
            length = math.hypot(x1 - x0, y1 - y0)
            # If the chain combined at least 2 dashes into a substantial edge:
            if length > 4.5:
                md["role"] = "hidden"
                reconstructed_hidden.append(md)
            else:
                # Still a short segment, keep as unknown
                reconstructed_hidden.append(md)
        lines = other_lines + reconstructed_hidden
    else:
        lines = other_lines + short_segs

    # 3.3 Detect Title Block and Border:
    # Title blocks are rectangular clusters in the corner with texts like "TITLE", "SCALE", "UNITS", "PROJECTION"
    all_pts = []
    for l in lines:
        all_pts += [l["p"][:2], l["p"][2:]]
    if all_pts:
        all_pts_arr = np.array(all_pts)
        min_x, min_y = all_pts_arr.min(0)
        max_x, max_y = all_pts_arr.max(0)
    else:
        min_x, min_y, max_x, max_y = 0, 0, 100, 100
        
    title_texts = [t for t in texts if any(k in t["text"].upper() for k in ("TITLE", "SCALE", "UNITS", "PROJECTION", "PART-"))]
    if title_texts:
        t_pts = np.array([t["at"] for t in title_texts])
        t_min_x, t_min_y = float(t_pts[:, 0].min() - 20), float(t_pts[:, 1].min() - 20)
        t_max_x, t_max_y = float(t_pts[:, 0].max() + 40), float(t_pts[:, 1].max() + 20)
        for l in lines:
            if l["role"] == "unknown":
                lx0, ly0, lx1, ly1 = l["p"]
                if (t_min_x <= lx0 <= t_max_x and t_min_y <= ly0 <= t_max_y and
                    t_min_x <= lx1 <= t_max_x and t_min_y <= ly1 <= t_max_y):
                    l["role"] = "title"
                    
    # 3.4 Detect Exploded Dimension Lines:
    # Look for arrowhead geometry (pairs of short segments at ~15 deg meeting at a point)
    # and lines that lie outside the main view envelopes
    # Any remaining unknown line with standard length is considered visible geometry
    for l in lines:
        if l["role"] == "unknown":
            l["role"] = "visible"
            
    # Clean circles and arcs
    for c in circles:
        c["role"] = "visible" if c.get("layer", "0") in ("VISIBLE", "0") else "unknown"
    for a in arcs:
        a["role"] = "visible" if a.get("layer", "0") in ("VISIBLE", "0") else "unknown"
        
    return {
        "insunits": ingested["insunits"],
        "units": ingested["units"],
        "lines": lines,
        "circles": circles,
        "arcs": arcs,
        "texts": texts,
        "dims": dims,
    }


# ---------------------------------------------------------------------------
# 4. Adaptive Endpoint Stitching
# ---------------------------------------------------------------------------
def stitch_endpoints(
    lines: List[Dict[str, Any]],
    user_tolerance: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Adaptive endpoint stitching.
    Clusters all segment endpoints within tolerance tau and snaps them to cluster centroids,
    closing drawing gaps caused by imperfect CAD tools, export rounding, or noise.
    Returns (stitched_lines, stitch_stats).
    """
    if not lines:
        return [], {"gaps_stitched": 0, "max_gap_closed": 0.0, "tolerance_used": 0.0}

    # Extract all endpoints
    pts = []
    for l in lines:
        pts.append(l["p"][:2])
        pts.append(l["p"][2:])
    pts_arr = np.array(pts)
    
    # Calculate drawing scale and adaptive tolerance
    min_xy = pts_arr.min(0)
    max_xy = pts_arr.max(0)
    diagonal = float(np.linalg.norm(max_xy - min_xy))
    
    # Adaptive tolerance: default to 0.005 * diagonal, clamped between [0.08, 0.60] mm
    if user_tolerance is not None:
        tau = user_tolerance
    else:
        tau = min(max(diagonal * 0.005, 0.08), 0.60)
        
    # Build KD-tree to cluster nearby endpoints
    tree = cKDTree(pts_arr)
    pairs = tree.query_pairs(r=tau)
    
    # Disjoint Set Union (DSU) to form connected endpoint clusters
    parent = list(range(len(pts_arr)))
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

    for i, j in pairs:
        union(i, j)
        
    clusters = defaultdict(list)
    for i in range(len(pts_arr)):
        clusters[find(i)].append(i)
        
    # Snap each cluster to its mean position
    snapped = np.copy(pts_arr)
    gaps_stitched = 0
    max_gap_closed = 0.0
    
    for root, idxs in clusters.items():
        if len(idxs) > 1:
            cluster_pts = pts_arr[idxs]
            centroid = cluster_pts.mean(axis=0)
            if len(idxs) == 2:
                gap = float(np.linalg.norm(cluster_pts[0] - cluster_pts[1]))
            else:
                gap = float(max(np.linalg.norm(p1 - p2) for p1 in cluster_pts for p2 in cluster_pts))
            max_gap_closed = max(max_gap_closed, gap)
            gaps_stitched += len(idxs) - 1
            for idx in idxs:
                snapped[idx] = centroid
                
    # Reconstruct stitched lines
    stitched_lines = []
    for idx, l in enumerate(lines):
        p0 = snapped[idx * 2]
        p1 = snapped[idx * 2 + 1]
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length > 1e-3:  # drop zero-length degenerates
            new_l = dict(l)
            new_l["p"] = [float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])]
            stitched_lines.append(new_l)
            
    stats = {
        "gaps_stitched": gaps_stitched,
        "max_gap_closed": round(max_gap_closed, 4),
        "tolerance_used": round(tau, 4),
        "total_endpoints": len(pts_arr),
        "clusters_formed": len(clusters),
    }
    return stitched_lines, stats


# ---------------------------------------------------------------------------
# 5. Closed Loop Construction (Shapely Polygonize)
# ---------------------------------------------------------------------------
def build_closed_loops(
    lines: List[Dict[str, Any]],
    arcs: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Polygon], List[Dict[str, Any]]]:
    """
    Construct planar closed polygons (faces/profiles with outer boundary and holes)
    from visible stitched lines and circular arcs.
    Returns (valid_polygons, unclosed_linestrings).
    """
    segments = []
    for l in lines:
        if l.get("role", "visible") == "visible":
            segments.append(LineString([l["p"][:2], l["p"][2:]]))
            
    if arcs:
        for a in arcs:
            if a.get("role", "visible") == "visible":
                c = a["c"]
                r = a["r"]
                a0, a1 = math.radians(a["a0"]), math.radians(a["a1"])
                sweep = (a1 - a0) % (2 * math.pi)
                n = max(8, int(sweep * r / 0.5) + 1)
                ang = a0 + np.linspace(0, sweep, n)
                pts = [[c[0] + r * math.cos(an), c[1] + r * math.sin(an)] for an in ang]
                segments.append(LineString(pts))
                
    # Unary union ensures clean node intersections
    merged_lines = unary_union(segments)
    
    # Polygonize finds all enclosed topological faces
    polys = list(polygonize(merged_lines))
    
    # Check validity and filter zero-area degenerates
    valid_polys = [p for p in polys if p.is_valid and p.area > 1.0]
    
    return valid_polys, segments


# ---------------------------------------------------------------------------
# 6. Dimension Extraction & Scale Regression
# ---------------------------------------------------------------------------
def regress_drawing_scale(
    ingested: Dict[str, Any],
    clean_lines: List[Dict[str, Any]],
    circles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Regress the drawing scale factor (scale = drawn_length / real_value).
    Cross-checks native dimensions and dimension texts against measured geometry.
    Detects standard scales: 1:1, 1:2, 2:1.
    """
    ratios = []
    details = []
    
    # 6.1 Native DIMENSION measurements
    for d in ingested.get("dims", []):
        meas = d["measurement"]
        txt = d["text"]
        # Extract numerical value from text override if present
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", txt)
        if m:
            val_nominal = float(m.group(0))
            if val_nominal > 0:
                ratio = meas / val_nominal
                ratios.append(ratio)
                details.append({"source": "native_dim_override", "meas": meas, "nominal": val_nominal, "ratio": ratio})
        elif meas > 0:
            details.append({"source": "native_dim_raw", "meas": meas, "nominal": meas, "ratio": 1.0})
            
    # 6.2 Check Title Block Text for Scale
    scale_from_title = None
    for t in ingested.get("texts", []):
        m = re.search(r"SCALE\s*([0-9]+)\s*[:/]\s*([0-9]+)", t["text"].upper())
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            if den > 0:
                scale_from_title = num / den
                
    if ratios:
        # Robust median ratio
        median_ratio = float(np.median(ratios))
        # Snap to nearest standard engineering scale
        standards = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0]
        closest_scale = min(standards, key=lambda s: abs(s - median_ratio))
        if abs(closest_scale - median_ratio) < 0.05:
            inferred_scale = closest_scale
        else:
            inferred_scale = median_ratio
    elif scale_from_title is not None:
        inferred_scale = scale_from_title
    else:
        inferred_scale = 1.0
        
    return {
        "scale": inferred_scale,
        "scale_text": f"1:{int(round(1.0 / inferred_scale))}" if inferred_scale < 1.0 else f"{int(round(inferred_scale))}:1",
        "scale_from_title": scale_from_title,
        "sample_count": len(ratios),
        "details": details
    }
