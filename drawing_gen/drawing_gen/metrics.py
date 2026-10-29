"""
Benchmark metrics.

edge_prf       point-sampled precision/recall/F1 between two entity sets (robust to fragmentation, duplicates,
               and primitive splitting, which are exactly what the noise injector does). Role-filterable.
volume_iou     exact B-rep volume IoU via OpenCascade booleans (use for the reconstruction's CadQuery solid).
hull_iou       voxel IoU between a solid and the VISUAL HULL of its three orthographic silhouettes
               (what Front ∩ Top ∩ Side would produce). Quantifies what hidden-line-only features cost.
flag_stats     self-flagging reliability: does status="needs_review" coincide with real failures?
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from project import sample


def _ent_points(e: dict, step: float) -> np.ndarray:
    g = e["g"]
    if e["kind"] == "line":
        return sample({"t": "line", "p": g["p"]}, step)
    if e["kind"] == "circle":
        return sample({"t": "circle", "c": g["c"], "r": g["r"]}, step)
    if e["kind"] == "arc":
        return sample({"t": "arc", "c": g["c"], "r": g["r"], "a0": g["a0"], "a1": g["a1"]}, step)
    return np.zeros((0, 2))


def edge_points(ents: list, roles=("visible",), step: float = 0.5) -> np.ndarray:
    pts = [_ent_points(e, step) for e in ents if e["role"] in roles and e["kind"] in ("line", "circle", "arc")]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def edge_prf(pred: list, gt: list, roles=("visible",), tol: float = 0.6, step: float = 0.5) -> dict:
    P, G = edge_points(pred, roles, step), edge_points(gt, roles, step)
    if len(P) == 0 or len(G) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prec = float((cKDTree(G).query(P)[0] <= tol).mean())
    rec = float((cKDTree(P).query(G)[0] <= tol).mean())
    return {"precision": prec, "recall": rec, "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)}


def volume_iou(a, b) -> float:
    """a, b: cq.Workplane or cq.Shape. Exact booleans; returns 0.0 if the boolean fails."""
    try:
        sa = a.val() if hasattr(a, "val") else a
        sb = b.val() if hasattr(b, "val") else b
        inter = sa.intersect(sb).Volume()
        union = sa.Volume() + sb.Volume() - inter
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


# ----------------------------------------------------------------------------- visual hull (voxel)
def _mesh(shape, tol=0.25):
    V, F = shape.tessellate(tol)
    return np.array([[v.x, v.y, v.z] for v in V]), np.array(F)


def _column_occupancy(V, F, xs, ys, zs):
    """Ray parity along +z for every (x, y) column -> boolean occupancy (nx, ny, nz)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d = np.array([0.0, 0.0, 1.0])
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    occ = np.zeros((len(xs), len(ys), len(zs)), bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            o = np.array([x, y, zs[0] - 5.0])
            f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
            s = o - p0
            u = f * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            v = f * (q @ d)
            t = f * np.einsum("ij,ij->i", e2, q)
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            ts = np.sort(t[hit])
            if len(ts) % 2:                                  # degenerate ray (edge hit): skip column
                continue
            for a0, a1 in zip(ts[0::2], ts[1::2]):
                occ[i, j, (zs - o[2] >= a0) & (zs - o[2] <= a1)] = True
    return occ


def _silhouette(V, F, ax_u, ax_v, lo, hi, n):
    """Rasterise projected triangles onto an n x n grid over [lo, hi] of the two axes (oversampled 4x)."""
    S = n * 4
    img = np.zeros((S, S), np.uint8)
    for tri in F:
        pts = V[tri][:, [ax_u, ax_v]]
        px = ((pts - lo[[ax_u, ax_v]]) / (hi[[ax_u, ax_v]] - lo[[ax_u, ax_v]]) * S).astype(np.int32)
        cv2.fillConvexPoly(img, px, 1)
    m = cv2.resize(img.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) > 0.5
    return m.T                                              # index [u, v]


def hull_iou(shape, n: int = 36) -> dict:
    """IoU between `shape` and the visual hull of its front/top/right silhouettes. hull ⊇ shape, so
    IoU = |shape| / |hull|."""
    V, F = _mesh(shape)
    lo, hi = V.min(0) - 1e-3, V.max(0) + 1e-3
    cx = lo[0] + (np.arange(n) + 0.5 + 0.0137) * (hi[0] - lo[0]) / n
    cy = lo[1] + (np.arange(n) + 0.5 + 0.0291) * (hi[1] - lo[1]) / n
    cz = lo[2] + (np.arange(n) + 0.5) * (hi[2] - lo[2]) / n
    truth = _column_occupancy(V, F, cx, cy, cz)
    fr = _silhouette(V, F, 0, 2, lo, hi, n)                 # front: (x, z)
    tp = _silhouette(V, F, 0, 1, lo, hi, n)                 # top:   (x, y)
    rt = _silhouette(V, F, 1, 2, lo, hi, n)                 # right: (y, z)
    hull = fr[:, None, :] & tp[:, :, None] & rt[None, :, :]
    inter, union = (truth & hull).sum(), (truth | hull).sum()
    return {"iou": float(inter / max(union, 1)), "truth_vox": int(truth.sum()), "hull_vox": int(hull.sum())}


# ----------------------------------------------------------------------------- self-flagging
def flag_stats(records: list, wrong_below: float = 0.90) -> dict:
    """records: [(volume_iou, flagged_bool)]. A result is 'wrong' if volume_iou < wrong_below.
    flag_recall   = P(flagged | wrong)      -> does it catch its own failures?
    false_alarm   = P(flagged | right)      -> does it cry wolf? (flagging everything would score recall = 1)"""
    wrong = [f for v, f in records if v < wrong_below]
    right = [f for v, f in records if v >= wrong_below]
    return {"n": len(records), "n_wrong": len(wrong), "n_right": len(right),
            "flag_recall": (sum(wrong) / len(wrong)) if wrong else float("nan"),
            "false_alarm": (sum(right) / len(right)) if right else float("nan"),
            "silent_failures": int(sum(1 for f in wrong if not f))}
