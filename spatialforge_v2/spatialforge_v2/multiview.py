"""
Multi-view sanity check. Silhouette IoU can be satisfied by the wrong 3D structure (e.g. a flat box
covering the right pixels). Render the model from other angles and ask the VLM whether it still
looks like the object. This is a *plausibility* signal, not proof; treat low scores as "needs review".
"""
from __future__ import annotations

import json
import re
from typing import Optional

import cv2
import numpy as np

from fit import Camera, Target, _project, make_frame
from features import feature_world_meshes
from geometry import part_mesh
from raster import rasterize, shade
from schema import Assembly
from solver import resolve

VIEWS = ((-35, 25), (45, 25), (135, 25), (0, 80))

SYSTEM = ("You review 3D reconstructions. You are shown renders of a 3D model, built from a sketch, seen from "
          "several angles. Judge structure only (parts, proportions, how parts connect), not surface detail.")


def scene_meshes(asm: Assembly, ov=None) -> tuple:
    solids = resolve(asm, ov)
    meshes = {p: part_mesh(s) for p, s in solids.items() if s.op == "union"}
    n_parts = len(meshes)
    for f in feature_world_meshes(asm, ov):
        if not f["is_cut"]:
            meshes[f["id"]] = f["mesh"]
    return meshes, solids, n_parts


def render_views(asm: Assembly, views=VIEWS, S: int = 220, ov=None) -> bytes:
    meshes, solids, n_parts = scene_meshes(asm, ov)
    tiles = []
    dummy = Target(np.ones((100, 100), np.uint8), S=S, margin=0.06)
    for az, el in views:
        cam = Camera(az, el)
        polys = _project(solids, cam)
        allp = np.vstack([p for p, _ in polys])
        wp, hp = allp.max(0) - allp.min(0)
        dummy.W, dummy.H = S * 0.88 * wp / max(wp, hp), S * 0.88 * hp / max(wp, hp)   # per-view aspect
        fr = make_frame(polys, dummy)
        ids, names, nb, _ = rasterize(meshes, cam, fr)
        tint = {k: (40, 130, 235) for k, n in enumerate(names) if k >= n_parts}
        img = shade(ids, nb, cam, tint, scale=1)
        cv2.putText(img, f"az {az} el {el}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        tiles.append(img)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 245))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    ok, buf = cv2.imencode(".png", np.vstack(rows))
    return buf.tobytes()


def build_prompt(asm: Assembly) -> str:
    return (f"The original object is described as: {asm.object_class}.\n"
            "Image 1: the original sketch. Image 2: four renders of the reconstructed 3D model.\n"
            "Does the model plausibly depict this object from every angle? Look for missing parts, extra parts, "
            "wrong attachments, parts floating or buried, or a body that is a flat slab where the sketch is clearly 3D.\n"
            'Return JSON only: {"plausible": true|false, "score": 0.0-1.0, "issues": ["<=5 short strings"]}')


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return {"plausible": bool(d.get("plausible")), "score": float(d.get("score", 0.0)),
                "issues": [str(x) for x in d.get("issues", [])][:5]}
    except Exception:
        return {"plausible": None, "score": None, "issues": ["could not parse the reviewer's answer"]}


def multiview_check(vlm, asm: Assembly, original_png: Optional[bytes] = None, ov=None) -> dict:
    imgs = ([original_png] if original_png else []) + [render_views(asm, ov=ov)]
    return parse(vlm.complete(SYSTEM, build_prompt(asm), imgs, None, 0.0))
