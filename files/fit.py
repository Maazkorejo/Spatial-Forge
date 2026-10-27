"""
Render-and-compare fitting loop  (Part 2 of 2)

Analysis-by-synthesis:
    1. search a coarse camera (az/el) for the candidate assembly
    2. render the assembly silhouette from that camera (orthographic, convex hull per part)
    3. score against the target mask: IoU + symmetric edge chamfer
    4. optimise the continuous dims (log space, bounded by each dim's [min, max]) and
       refine the camera, with a weak prior toward the VLM's original values
    5. measure per-dimension sensitivity so unconstrained dims get labelled "assumed"

Known limits (be honest in the UI):
    * orthographic camera; perspective sketches are approximated
    * difference (cut) parts do not affect the silhouette, so they are not fitted
    * one silhouette cannot fix the global scale or hidden depth; sensitivity flags this
    * for true occlusion/edge fitting swap `render` for a depth-buffered renderer
      (pyrender / nvdiffrast) and add an edge-map term. The interface stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize

from schema import Assembly, ViewHint
from solver import resolve, surface_points

K3 = np.ones((3, 3), np.uint8)


# --------------------------------------------------------------------------- camera
@dataclass
class Camera:
    az: float   # degrees, 0 = looking at the front (+y direction)
    el: float   # degrees above the horizon


def basis(az: float, el: float):
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])  # toward camera
    right = np.array([np.cos(a), np.sin(a), 0.0])
    up = np.cross(d, right)
    return right, up


# --------------------------------------------------------------------------- target
class Target:
    """Binary silhouette normalised into an S x S canvas (aspect preserved)."""

    def __init__(self, mask: np.ndarray, S: int = 160, margin: float = 0.08):
        m = (mask > 0).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            raise ValueError("empty target mask")
        crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
        h, w = crop.shape
        sc = S * (1 - 2 * margin) / max(h, w)
        nh, nw = max(1, round(h * sc)), max(1, round(w * sc))
        crop = (cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        self.S = S
        self.mask = np.zeros((S, S), np.uint8)
        y0, x0 = (S - nh) // 2, (S - nw) // 2
        self.mask[y0:y0 + nh, x0:x0 + nw] = crop
        self.cx, self.cy = x0 + nw / 2, y0 + nh / 2
        self.W, self.H = float(nw), float(nh)
        edge = self.mask - cv2.erode(self.mask, K3)
        self.edge = edge
        self.dt_edge = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3) / S


# --------------------------------------------------------------------------- render
def _project(solids, cam: Camera):
    right, up = basis(cam.az, cam.el)
    polys = []
    for s in solids.values():
        if s.op != "union":
            continue
        p = surface_points(s)
        polys.append(np.stack([p @ right, p @ up], axis=1))
    return polys


def render(asm: Assembly, cam: Camera, target: Target, ov=None) -> Optional[np.ndarray]:
    polys = _project(resolve(asm, ov), cam)
    allp = np.vstack(polys)
    lo, hi = allp.min(0), allp.max(0)
    wp, hp = hi - lo
    if wp < 1e-6 or hp < 1e-6:
        return None
    # similarity transform (uniform scale + centring) that best matches the target extents
    s = (target.W * wp + target.H * hp) / (wp * wp + hp * hp)
    ctr = (lo + hi) / 2
    img = np.zeros((target.S, target.S), np.uint8)
    for uv in polys:
        xy = np.empty_like(uv)
        xy[:, 0] = target.cx + (uv[:, 0] - ctr[0]) * s
        xy[:, 1] = target.cy - (uv[:, 1] - ctr[1]) * s
        hull = cv2.convexHull(np.round(xy).astype(np.int32))
        cv2.fillConvexPoly(img, hull, 1)
    return img


def score(img: np.ndarray, target: Target) -> tuple[float, float]:
    inter = int((img & target.mask).sum())
    union = int((img | target.mask).sum())
    iou = inter / max(union, 1)
    edge = img - cv2.erode(img, K3)
    if edge.sum() == 0:
        return iou, 1.0
    dt_r = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3) / target.S
    c1 = float(target.dt_edge[edge > 0].mean())      # render edge -> target edge
    c2 = float(dt_r[target.edge > 0].mean())         # target edge -> render edge
    return iou, (c1 + c2) / 2


def diagnose(img: np.ndarray, target: Target) -> dict:
    """Where does the render disagree? Coordinates are normalised (0..1, origin top-left)."""
    fn = (target.mask == 1) & (img == 0)     # sketch silhouette not explained by the model
    fp = (img == 1) & (target.mask == 0)     # model sticks out of the sketch silhouette
    tot = max(int(target.mask.sum()), 1)

    def cen(m):
        ys, xs = np.nonzero(m)
        return None if len(xs) < 0.01 * tot else [round(float(xs.mean() / target.S), 2),
                                                  round(float(ys.mean() / target.S), 2)]

    return {"uncovered_frac": round(float(fn.sum() / tot), 3), "uncovered_center": cen(fn),
            "excess_frac": round(float(fp.sum() / tot), 3), "excess_center": cen(fp)}


def make_overlay(img: np.ndarray, target: Target, scale: int = 3) -> bytes:
    """green = sketch silhouette only, red = model only, yellow = both."""
    rgb = np.zeros((target.S, target.S, 3), np.uint8)
    t, r = target.mask == 1, img == 1
    rgb[t & ~r] = (0, 200, 0)
    rgb[r & ~t] = (0, 0, 220)
    rgb[t & r] = (0, 220, 220)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", rgb)
    return buf.tobytes()


# --------------------------------------------------------------------------- camera search
def search_camera(asm: Assembly, target: Target, hint: Optional[ViewHint] = None) -> Camera:
    hint = hint or asm.view_hint
    fixed = {"orthographic_front": (0, 0), "orthographic_side": (90, 0), "orthographic_top": (0, 89)}
    if hint.projection in fixed:
        return Camera(*fixed[hint.projection])
    if hint.azimuth_deg is not None and hint.elevation_deg is not None:
        azs = [hint.azimuth_deg + d for d in (-20, -10, 0, 10, 20)]
        els = [max(2, hint.elevation_deg + d) for d in (-15, 0, 15)]
    else:
        azs = list(range(-165, 180, 15))
        els = [10, 20, 30, 40, 50]
    best, best_s = Camera(35, 25), -1.0
    for az in azs:
        for el in els:
            img = render(asm, Camera(az, el), target)
            if img is None:
                continue
            iou, ch = score(img, target)
            if iou - ch > best_s:
                best, best_s = Camera(az, el), iou - ch
    return best


# --------------------------------------------------------------------------- fit
@dataclass
class FitResult:
    iou: float
    init_iou: float
    camera: Camera
    values: dict                       # {(part_id, field): value}
    sens_slot: dict                    # {(part_id, field): loss increase when perturbed +-15%}
    sens_part: dict                    # {part_id: max sensitivity}
    rendered: Optional[np.ndarray] = None
    n_evals: int = 0
    notes: list = field(default_factory=list)


def fit(asm: Assembly, target: Target, cam: Optional[Camera] = None, fit_camera: bool = True,
        lam: float = 0.02, maxiter: int = 30, popsize: int = 8, seed: int = 0,
        polish: bool = True) -> FitResult:
    slots = asm.dim_slots()
    n = len(slots)
    bounds, x0 = [], []
    for _, _, d in slots:
        a, b = (d.value * 0.98, d.value * 1.02) if d.source == "measured" else (d.min, d.max)
        a, b = np.log(a), np.log(b)
        if b - a < 1e-6:
            a, b = a - 1e-6, b + 1e-6
        bounds.append((a, b))
        x0.append(np.log(d.value))
    cam = cam or search_camera(asm, target)
    if fit_camera:
        bounds += [(cam.az - 20, cam.az + 20), (max(2.0, cam.el - 15), min(85.0, cam.el + 15))]
        x0 += [cam.az, cam.el]
    x0 = np.array(x0)
    log0 = x0[:n].copy()
    counter = {"n": 0}

    def unpack(x):
        ov = {(pid, f): float(np.exp(x[i])) for i, (pid, f, _) in enumerate(slots)}
        c = Camera(float(x[n]), float(x[n + 1])) if fit_camera else cam
        return ov, c

    def base_loss(x):
        counter["n"] += 1
        ov, c = unpack(x)
        img = render(asm, c, target, ov)
        if img is None:
            return 10.0, None
        iou, ch = score(img, target)
        return (1.0 - iou) + 2.0 * ch, img

    def objective(x):
        l, _ = base_loss(x)
        return l + lam * float(np.mean((x[:n] - log0) ** 2))

    _, img0 = base_loss(x0)
    init_iou = score(img0, target)[0] if img0 is not None else 0.0

    res = differential_evolution(objective, bounds, x0=x0, maxiter=maxiter, popsize=popsize,
                                 tol=1e-4, seed=seed, polish=False, init="sobol",
                                 mutation=(0.5, 1.0), recombination=0.8)
    x = res.x
    if polish:
        try:
            r2 = minimize(objective, x, method="Powell", bounds=bounds,
                          options={"maxiter": 400, "xtol": 1e-3, "ftol": 1e-5})
            if r2.fun <= res.fun:
                x = r2.x
        except Exception:
            pass

    base, img = base_loss(x)
    iou = score(img, target)[0] if img is not None else 0.0

    # sensitivity: how much does the unregularised loss rise when this dim moves +-15%?
    sens_slot, sens_part = {}, {}
    for i, (pid, f, _) in enumerate(slots):
        worst = 0.0
        for fac in (0.85, 1.15):
            xp = x.copy()
            xp[i] = np.clip(xp[i] + np.log(fac), bounds[i][0] - 0.2, bounds[i][1] + 0.2)
            worst = max(worst, base_loss(xp)[0] - base)
        sens_slot[(pid, f)] = worst
        sens_part[pid] = max(sens_part.get(pid, 0.0), worst)

    ov, c = unpack(x)
    notes = []
    if iou < 0.8:
        notes.append("low silhouette agreement: structure is probably wrong, not just the dims")
    return FitResult(iou, init_iou, c, ov, sens_slot, sens_part, img, counter["n"], notes)


SENS_THRESHOLD = 0.004


def apply_fit(asm: Assembly, fr: FitResult) -> Assembly:
    """Return a new Assembly with fitted values and honest provenance labels."""
    out = asm.model_copy(deep=True)
    for p in out.parts:
        for f, d in p.dims().items():
            key = (p.id, f)
            if key in fr.values:
                d.value = float(np.clip(fr.values[key], d.min, d.max))
                if d.source != "measured":
                    d.source = "inferred" if fr.sens_slot.get(key, 0) >= SENS_THRESHOLD else "assumed"
            elif d.source != "measured":
                d.source = "assumed"          # cut parts: not constrained by the silhouette
    out.view_hint.azimuth_deg = round(fr.camera.az, 1)
    out.view_hint.elevation_deg = round(fr.camera.el, 1)
    return out
