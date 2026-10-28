"""
Small numpy z-buffer rasterizer (orthographic). Not for the optimisation inner loop
(fit.py uses cheap polygon fills there); used for:
  * per-part visibility masks -> per-part IoU against SAM part masks
  * shaded views for the UI overlay and the multi-view sanity check
For speed or perspective, swap in pyrender / nvdiffrast behind the same function signatures.
"""
from __future__ import annotations

import cv2
import numpy as np

from fit import Camera, Frame, basis


def view_dir(cam: Camera) -> np.ndarray:
    a, e = np.radians(cam.az), np.radians(cam.el)
    return np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])


def rasterize(meshes: dict, cam: Camera, frame: Frame):
    """meshes: {name: (V, F)}. Returns (ids (S,S) int, names list, normals (S,S,3), depth (S,S))."""
    S = frame.S
    right, up = basis(cam.az, cam.el)
    d = view_dir(cam)
    zbuf = np.full((S, S), -np.inf)
    ids = np.full((S, S), -1, np.int32)
    nbuf = np.zeros((S, S, 3), np.float32)
    names = list(meshes)
    for k, name in enumerate(names):
        V, F = meshes[name]
        px = frame.to_px(np.stack([V @ right, V @ up], axis=1))
        depth = V @ d
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        for t in np.nonzero(n @ d > 1e-6)[0]:           # front-facing only (meshes are closed)
            (x0, y0), (x1, y1), (x2, y2) = px[F[t]]
            xa, xb = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))), S - 1)
            ya, yb = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))), S - 1)
            if xb < xa or yb < ya:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            X, Y = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
            l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
            l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            z = l0 * depth[F[t, 0]] + l1 * depth[F[t, 1]] + l2 * depth[F[t, 2]]
            zs = zbuf[ya:yb + 1, xa:xb + 1]
            upd = inside & (z > zs)
            if upd.any():
                zs[upd] = z[upd]
                ids[ya:yb + 1, xa:xb + 1][upd] = k
                nbuf[ya:yb + 1, xa:xb + 1][upd] = n[t]
    return ids, names, nbuf, zbuf


def part_visibility_iou(ids: np.ndarray, names: list, part_masks: dict) -> dict:
    """IoU between each part's visible region and a given mask (canvas coords). Missing => 0."""
    out = {}
    for pid, m in part_masks.items():
        if pid not in names:
            out[pid] = 0.0
            continue
        r = ids == names.index(pid)
        m = m > 0
        out[pid] = float((r & m).sum() / max((r | m).sum(), 1))
    return out


def shade(ids: np.ndarray, nbuf: np.ndarray, cam: Camera, tint: dict = None, scale: int = 2) -> np.ndarray:
    """Lambert + edge lines -> BGR image. tint: {index: (b, g, r)} base colour per mesh index."""
    right, up = basis(cam.az, cam.el)
    d = view_dir(cam)
    L = d + 0.6 * up - 0.4 * right
    L /= np.linalg.norm(L)
    lam = np.clip(nbuf @ L, 0, 1)
    img = np.full(ids.shape + (3,), 245, np.float32)
    mask = ids >= 0
    base = np.full(ids.shape + (3,), 190, np.float32)
    if tint:
        for k, col in tint.items():
            base[ids == k] = col
    img[mask] = (base * (0.35 + 0.65 * lam[..., None]))[mask]
    # edges where the id or the normal changes
    edge = np.zeros(ids.shape, bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[1:, :] |= ids[1:, :] != ids[:-1, :]
    dn = np.abs(nbuf[:, 1:] - nbuf[:, :-1]).sum(-1) > 0.8
    edge[:, 1:] |= dn & mask[:, 1:]
    dn = np.abs(nbuf[1:, :] - nbuf[:-1, :]).sum(-1) > 0.8
    edge[1:, :] |= dn & mask[1:, :]
    img[edge] = (40, 40, 40)
    img = img.astype(np.uint8)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
