"""
Mesh generation for every primitive + the small feature library.

All meshes are (V, F): V (n,3) float, F (m,3) int, outward-facing, watertight.
Convex primitives are meshed as convex hulls of surface samples; non-convex extrusion
profiles use ear clipping (no external triangulation dependency).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from solver import Solid, surface_points

Mesh = tuple  # (V, F)


# ----------------------------------------------------------------------------- triangulation
def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_tri(p, a, b, c, eps=1e-12):
    return _area2(a, b, p) >= -eps and _area2(b, c, p) >= -eps and _area2(c, a, p) >= -eps


def earclip(poly: np.ndarray) -> list:
    """Triangulate a simple CCW polygon (n,2). Returns index triples."""
    n = len(poly)
    idx = list(range(n))
    tris = []
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 4 * n * n:
            raise ValueError("earclip failed: polygon is not simple / not CCW")
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if _area2(a, b, c) <= 1e-12:
                continue
            if any(_in_tri(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:                       # drop a collinear vertex and retry
            for k in range(len(idx)):
                i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
                if abs(_area2(poly[i0], poly[i1], poly[i2])) <= 1e-12:
                    idx.pop(k)
                    clipped = True
                    break
            if not clipped:
                raise ValueError("earclip failed: no ear found")
    tris.append(tuple(idx))
    return tris


def prism(poly_xy: np.ndarray, z0: float, z1: float) -> Mesh:
    """Extrude a CCW polygon (n,2) along +z from z0 to z1."""
    poly = np.asarray(poly_xy, float)
    n = len(poly)
    V = np.vstack([np.c_[poly, np.full(n, z0)], np.c_[poly, np.full(n, z1)]])
    F = []
    for a, b, c in earclip(poly):
        F.append((n + a, n + b, n + c))        # top cap, normal +z
        F.append((a, c, b))                    # bottom cap, normal -z
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, n + j))
        F.append((i, n + j, n + i))
    return V, np.array(F, int)


def hull_mesh(points: np.ndarray) -> Mesh:
    h = ConvexHull(points)
    V, F = points[h.vertices], None
    remap = {v: i for i, v in enumerate(h.vertices)}
    F = np.array([[remap[v] for v in tri] for tri in h.simplices])
    # orient outward using hull equations
    for k, tri in enumerate(h.simplices):
        a, b, c = points[tri]
        if np.dot(np.cross(b - a, c - a), h.equations[k, :3]) < 0:
            F[k] = F[k][[0, 2, 1]]
    return V, F


# ----------------------------------------------------------------------------- primitives
def rounded_box_points(center, size, r, n=4) -> np.ndarray:
    r = float(min(r, 0.49 * np.min(size)))
    h = np.asarray(size) / 2 - r
    t = np.linspace(0, np.pi / 2, n + 1)
    dirs = np.array([[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)] for a in t for b in t])
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                sg = np.array([sx, sy, sz])
                pts.append(center + sg * (h + r * dirs))
    return np.vstack(pts)


def sphere_points(center, size, n=10) -> np.ndarray:
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    P = np.array([[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)] for a in th for b in ph])
    return center + P * (np.asarray(size) / 2)


def hull_points(s: Solid) -> np.ndarray:
    """Points whose hull is the part's (convex) shape. Used for silhouettes and meshes."""
    if s.kind == "rounded_box":
        return rounded_box_points(s.center, s.size, s.fillet)
    if s.kind == "sphere":
        return sphere_points(s.center, s.size)
    return surface_points(s, n=32)


def extrusion_profile_xz(s: Solid) -> np.ndarray:
    """World (x, z) vertices of the profile."""
    p = np.asarray(s.profile, float)
    lo = s.center - s.size / 2
    return np.c_[lo[0] + p[:, 0] * s.size[0], lo[2] + p[:, 1] * s.size[2]]


def part_mesh(s: Solid) -> Mesh:
    if s.kind == "extrusion":
        xz = extrusion_profile_xz(s)
        d = s.size[1]
        V, F = prism(xz - np.array([0.0, 0.0]), 0.0, d)      # local (x, z_world) -> prism xy, depth along local z
        # local (lx, ly, lz) -> world (lx, -lz, ly); then shift depth so it is centred on s.center[1]
        W = np.c_[V[:, 0], -V[:, 2], V[:, 1]]
        W[:, 1] += s.center[1] + d / 2
        return W, F
    return hull_mesh(hull_points(s))


# ----------------------------------------------------------------------------- silhouettes
def silhouette_polys(s: Solid) -> list:
    """List of (points3d, is_convex). Filling all of them (hull for convex, fillPoly otherwise) gives the silhouette."""
    if s.kind != "extrusion":
        return [(hull_points(s), True)]
    xz = extrusion_profile_xz(s)
    y0, y1 = s.center[1] - s.size[1] / 2, s.center[1] + s.size[1] / 2
    out = []
    for y in (y0, y1):
        out.append((np.c_[xz[:, 0], np.full(len(xz), y), xz[:, 1]], False))
    n = len(xz)
    for i in range(n):
        j = (i + 1) % n
        q = np.array([[xz[i, 0], y0, xz[i, 1]], [xz[j, 0], y0, xz[j, 1]],
                      [xz[j, 0], y1, xz[j, 1]], [xz[i, 0], y1, xz[i, 1]]])
        out.append((q, True))
    return out


# ----------------------------------------------------------------------------- feature library
def _circle(d, n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)] * (d / 2)


def feature_outline(kind: str, dims: dict) -> np.ndarray:
    """CCW 2D outline of the feature's footprint in the face tangent plane."""
    if kind in ("button", "hole", "screw"):
        return _circle(dims["diameter"])
    if kind == "dial":                 # fluted knob: every third vertex pulled in
        n = 36
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = (dims["diameter"] / 2) * np.where((np.arange(n) % 3) == 0, 0.9, 1.0)
        return np.c_[np.cos(t) * r, np.sin(t) * r]
    a, b = dims["size_u"] / 2, dims["size_v"] / 2
    return np.array([[-a, -b], [a, -b], [a, b], [-a, b]])


def feature_mesh_local(kind: str, dims: dict, cutter: bool = False) -> Mesh:
    """Mesh in the feature frame: origin on the surface, +z outward, (x, y) = face tangents.
    cutter=True returns the volume to SUBTRACT (extends from -depth to a small overshoot)."""
    depth = dims["depth"]
    z0, z1 = (-depth, 0.5) if cutter else (-0.3, depth)   # tiny embed so unions stay watertight
    return prism(feature_outline(kind, dims), z0, z1)


def transform(V: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return V @ R.T + origin
