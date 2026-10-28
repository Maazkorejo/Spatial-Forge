"""Synthetic test cases: render a ground-truth assembly as a fake 'sketch' (mask + feature detections).
No VLM needed. Lets us measure the fit, feature placement and exports against known truth."""
from __future__ import annotations

import cv2
import numpy as np

import fit as F
from features import Detection, feature_world_meshes
from schema import Assembly
from solver import resolve


def perturb(truth: Assembly, rng, lo=0.8, hi=1.25, tol=0.4) -> Assembly:
    """Simulate a VLM guess: wrong dims (within lo..hi) with ranges re-centred on the guess."""
    a = truth.model_copy(deep=True)
    for p in a.parts:
        for _, d in p.dims().items():
            v = d.value * rng.uniform(lo, hi)
            d.value, d.min, d.max = v, v * (1 - tol), v * (1 + tol)
            d.source = "inferred"
    return a


def synthetic_case(truth: Assembly, cam: F.Camera, W: int = 900, H: int = 700, margin: float = 0.12):
    solids = resolve(truth)
    polys = F._project(solids, cam)
    allp = np.vstack([p for p, _ in polys])
    lo, hi = allp.min(0), allp.max(0)
    s0 = min(W * (1 - 2 * margin) / (hi - lo)[0], H * (1 - 2 * margin) / (hi - lo)[1])
    c = (lo + hi) / 2
    right, up = F.basis(cam.az, cam.el)

    def to_px(uv):
        return np.c_[W / 2 + (uv[:, 0] - c[0]) * s0, H / 2 - (uv[:, 1] - c[1]) * s0]

    mask = np.zeros((H, W), np.uint8)
    for uv, convex in polys:
        pts = np.round(to_px(uv)).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255) if convex else cv2.fillPoly(mask, [pts], 255)

    dets = []
    for f in feature_world_meshes(truth):
        if f["is_cut"]:
            continue
        V = f["mesh"][0]
        px = to_px(np.c_[V @ right, V @ up])
        x0, y0 = px.min(0)
        x1, y1 = px.max(0)
        dets.append(Detection(label=f["kind"], box=(x0, y0, x1, y1), score=0.9))
    # cut features: detector sees the opening footprint
    for f in feature_world_meshes(truth):
        if f["is_cut"]:
            V = f["cutter"][0]
            V = V[(V - f["origin"]) @ f["normal"] > -1e-6]          # opening footprint only
            px = to_px(np.c_[V @ right, V @ up])
            (x0, y0), (x1, y1) = px.min(0), px.max(0)
            dets.append(Detection(label=f["kind"], box=(x0, y0, x1, y1), score=0.8))
    image = np.dstack([255 - mask] * 3)
    return {"image": image, "mask": mask, "detections": dets, "to_px": to_px, "s0": s0}
