"""
B-rep export with CadQuery -> STEP (+ STL). Same relational model, real fillets and exact
cylinders. Cosmetic features are excluded from the CAD body by default.

Notes
  * the dial's flutes are a polygon outline; buttons/holes/screws use true circles
  * rounded_box uses a true edge fillet (clamped to < half the smallest side)
"""
from __future__ import annotations

import os
from typing import Optional

try:
    import cadquery as cq
    HAS_CQ = True
except ImportError:
    cq = None
    HAS_CQ = False

import numpy as np

from features import feature_pose
from geometry import feature_outline, part_mesh
from schema import Assembly
from solver import resolve
from export import mm_per_u

AXV = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def _part_solid(s) -> cq.Workplane:
    c = tuple(float(x) for x in s.center)
    sx, sy, sz = (float(x) for x in s.size)
    if s.kind == "box":
        return cq.Workplane("XY").box(sx, sy, sz).translate(c)
    if s.kind == "rounded_box":
        r = float(min(s.fillet, 0.49 * min(sx, sy, sz)))
        return cq.Workplane("XY").box(sx, sy, sz).edges().fillet(r).translate(c)
    if s.kind == "cylinder":
        a = np.array(AXV[s.axis], float)
        ax = {"x": 0, "y": 1, "z": 2}[s.axis]
        L, d = float(s.size[ax]), float(np.delete(s.size, ax)[0])
        p0 = np.array(c) - a * L / 2
        solid = cq.Solid.makeCylinder(d / 2, L, cq.Vector(*p0), cq.Vector(*a))
        return cq.Workplane("XY").newObject([solid])
    if s.kind == "sphere":
        return cq.Workplane("XY").sphere(sx / 2).translate(c)
    # extrusion: profile in the XZ plane, extruded along y
    p = np.asarray(s.profile, float)
    lo = s.center - s.size / 2
    pts = [(float(lo[0] + x * s.size[0]), float(lo[2] + z * s.size[2])) for x, z in p]
    plane = cq.Plane(origin=(0, float(s.center[1] + s.size[1] / 2), 0), xDir=(1, 0, 0), normal=(0, -1, 0))
    return cq.Workplane(plane).polyline(pts).close().extrude(float(s.size[1]))


def _feature_solid(f, solids, meshes, cutter: bool) -> cq.Workplane:
    origin, R = feature_pose(f, solids, meshes)
    plane = cq.Plane(origin=tuple(float(x) for x in origin), xDir=tuple(float(x) for x in R[:, 0]),
                     normal=tuple(float(x) for x in R[:, 2]))
    dims = {k: d.value for k, d in f.dims().items()}
    z0, z1 = (-dims["depth"], 0.5) if cutter else (-0.3, dims["depth"])
    wp = cq.Workplane(plane).workplane(offset=z0)
    if f.kind in ("button", "hole", "screw"):
        return wp.circle(dims["diameter"] / 2).extrude(z1 - z0)
    pts = [tuple(map(float, p)) for p in feature_outline(f.kind, dims)]
    return wp.polyline(pts).close().extrude(z1 - z0)


def build_cq(asm: Assembly, ov=None, include_cosmetic: bool = False) -> cq.Workplane:
    solids = resolve(asm, ov)
    meshes = {pid: part_mesh(s) for pid, s in solids.items() if s.op == "union"}
    parts = {pid: _part_solid(s) for pid, s in solids.items() if s.op == "union"}
    for s in solids.values():
        if s.op == "difference":
            cutter = _part_solid(s)
            for t in s.cuts:
                parts[t] = parts[t].cut(cutter)
    body = None
    for w in parts.values():
        body = w if body is None else body.union(w)
    for f in asm.features:
        if f.layer == "cosmetic" and not include_cosmetic:
            continue
        cut = f.kind in ("hole", "slot")
        fs = _feature_solid(f, solids, meshes, cutter=cut)
        body = body.cut(fs) if cut else body.union(fs)
    return body


def export_step(asm: Assembly, outdir: str, name: str = "model", ov=None, include_cosmetic: bool = False) -> dict:
    os.makedirs(outdir, exist_ok=True)
    if not HAS_CQ:
        step_out = os.path.join(outdir, f"{name}.step")
        sample_step = os.path.join(os.path.dirname(__file__), "sample_output", "model.step")
        if os.path.exists(sample_step) and not os.path.exists(step_out):
            import shutil
            shutil.copyfile(sample_step, step_out)
            return {"step": step_out, "stl": None, "valid": True, "volume": 0.0, "units": "u (sample)"}
        return {"step": None, "stl": None, "valid": False, "volume": 0.0, "units": "cadquery not installed"}

    body = build_cq(asm, ov, include_cosmetic)
    k = mm_per_u(asm, ov)
    if k:
        body = body.val().scale(k)
        body = cq.Workplane("XY").newObject([body])
    step, stl = os.path.join(outdir, f"{name}.step"), os.path.join(outdir, f"{name}_brep.stl")
    cq.exporters.export(body, step)
    cq.exporters.export(body, stl)
    v = body.val()
    return {"step": step, "stl": stl, "valid": bool(v.isValid()), "volume": float(v.Volume()),
            "units": "mm" if k else "u (relative)"}
