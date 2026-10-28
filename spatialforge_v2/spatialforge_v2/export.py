"""
Mesh export: manifold3d booleans -> watertight STL (CAD layer) and GLB (visual layer).

Layers
  * CAD body   : union parts - cuts, + protruding features, - hole/slot cutters. Cosmetic excluded.
  * Visual GLB : same body, plus cosmetic features, with per-layer colours.
Scale: millimetres only if the assembly has an anchor (printed/user dimension); otherwise
files stay in relative units "u" and `info["scale"]` says so. Mass is reported only when scaled.
"""
from __future__ import annotations

import json
import os
from typing import Optional

try:
    import manifold3d as m3d
    HAS_M3D = True
except ImportError:
    m3d = None
    HAS_M3D = False

import numpy as np
import trimesh

from features import feature_world_meshes
from geometry import part_mesh
from schema import Assembly
from solver import resolve

AL6061_G_PER_CM3 = 2.70


def _mf(mesh) -> "m3d.Manifold":
    V, F = mesh
    return m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(F, np.uint32)))


def _to_trimesh(m: "m3d.Manifold") -> trimesh.Trimesh:
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def mm_per_u(asm: Assembly, ov=None) -> Optional[float]:
    if asm.anchor is None:
        return None
    p = next(p for p in asm.parts if p.id == asm.anchor.part)
    v = (ov or {}).get((p.id, asm.anchor.dim), getattr(p, asm.anchor.dim).value)
    return asm.anchor.mm / v


def build_body(asm: Assembly, ov=None, include_cosmetic: bool = False) -> trimesh.Trimesh:
    solids = resolve(asm, ov)
    if HAS_M3D:
        parts = {pid: _mf(part_mesh(s)) for pid, s in solids.items() if s.op == "union"}
        for s in solids.values():
            if s.op == "difference":
                cutter = _mf(part_mesh(s))
                for t in s.cuts:
                    parts[t] = parts[t] - cutter
        body = None
        for m in parts.values():
            body = m if body is None else body + m
        for f in feature_world_meshes(asm, ov):
            if f["layer"] == "cosmetic" and not include_cosmetic:
                continue
            body = body - _mf(f["cutter"]) if f["is_cut"] else body + _mf(f["mesh"])
        return _to_trimesh(body)
    else:
        meshes = []
        for pid, s in solids.items():
            if s.op == "union":
                V, F = part_mesh(s)
                meshes.append(trimesh.Trimesh(V, F, process=False))
        for f in feature_world_meshes(asm, ov):
            if f["layer"] == "cosmetic" and not include_cosmetic:
                continue
            if not f["is_cut"]:
                V, F = f["mesh"]
                meshes.append(trimesh.Trimesh(V, F, process=False))
        if meshes:
            return trimesh.util.concatenate(meshes)
        return trimesh.Trimesh()


def export_all(asm: Assembly, outdir: str, name: str = "model", ov=None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    k = mm_per_u(asm, ov)
    scale = k if k else 1.0
    cad = build_body(asm, ov, include_cosmetic=False)
    cad.apply_scale(scale)
    stl = os.path.join(outdir, f"{name}.stl")
    cad.export(stl)

    # visual scene
    scene = trimesh.Scene()
    vis = build_body(asm, ov, include_cosmetic=True)
    vis.apply_scale(scale)
    vis.visual.face_colors = (200, 200, 205, 255)
    scene.add_geometry(vis, node_name="body")
    for f in feature_world_meshes(asm, ov):
        if f["is_cut"]:
            continue
        V, F = f["mesh"]
        t = trimesh.Trimesh(V * scale, F, process=False)
        t.visual.face_colors = (230, 120, 30, 255) if f["layer"] == "feature" else (90, 90, 95, 255)
        scene.add_geometry(t, node_name=f["id"])
    glb = os.path.join(outdir, f"{name}.glb")
    scene.export(glb)

    info = {
        "stl": stl, "glb": glb,
        "units": "mm" if k else "u (relative, no anchor dimension)",
        "scale": k, "watertight": bool(cad.is_watertight),
        "volume": float(cad.volume), "n_triangles": int(len(cad.faces)),
        "mass_g_al6061": float(cad.volume / 1000 * AL6061_G_PER_CM3) if k else None,
        "note": "Mass is shown only when an anchor dimension fixes the scale.",
    }
    with open(os.path.join(outdir, f"{name}.json"), "w") as fh:
        fh.write(asm.model_dump_json(exclude_none=True, indent=2))
    return info
