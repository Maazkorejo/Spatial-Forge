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
from geometry import silhouette_polys
from solver import resolve

K3 = np.ones((3, 3), np.uint8)


# --------------------------------------------------------------------------- camera
@dataclass
class Camera:
    az: float   # degrees, 0 = looking at the front (+y direction)
    el: float   # degrees above the horizon
    roll: float = 0.0  # degrees of in-plane camera roll


def basis(az: float, el: float, roll: float = 0.0):
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])  # toward camera
    r0 = np.array([np.cos(a), np.sin(a), 0.0])
    u0 = np.cross(d, r0)
    if roll != 0.0:
        psi = np.radians(roll)
        c, s = np.cos(psi), np.sin(psi)
        right = c * r0 + s * u0
        up = -s * r0 + c * u0
    else:
        right, up = r0, u0
    return right, up


# --------------------------------------------------------------------------- target
class Target:
    """Binary silhouette normalised into an S x S canvas (aspect preserved).
    Keeps the source->canvas transform so detections / part masks given in ORIGINAL image
    pixels can be mapped into the canvas (to_canvas / warp)."""

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
        # source pixel -> canvas pixel:  c = (p - src_min) * k + off
        self.kx, self.ky = nw / w, nh / h
        self.src_min = (float(xs.min()), float(ys.min()))
        self.off = (float(x0), float(y0))
        edge = self.mask - cv2.erode(self.mask, K3)
        self.edge = edge
        self.dt_edge = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3) / S

    def to_canvas(self, x: float, y: float) -> tuple:
        return ((x - self.src_min[0]) * self.kx + self.off[0], (y - self.src_min[1]) * self.ky + self.off[1])

    def warp(self, full_mask: np.ndarray) -> np.ndarray:
        """Map a binary mask in original-image coordinates into the canvas."""
        M = np.array([[self.kx, 0, self.off[0] - self.src_min[0] * self.kx],
                      [0, self.ky, self.off[1] - self.src_min[1] * self.ky]], np.float32)
        out = cv2.warpAffine((full_mask > 0).astype(np.uint8), M, (self.S, self.S), flags=cv2.INTER_NEAREST)
        return out


# --------------------------------------------------------------------------- render
def _project(solids, cam: Camera):
    """-> list of (uv (k,2), is_convex) for every union solid."""
    right, up = basis(cam.az, cam.el, getattr(cam, "roll", 0.0))
    polys = []
    for s in solids.values():
        if s.op != "union":
            continue
        for p3, convex in silhouette_polys(s):
            polys.append((np.stack([p3 @ right, p3 @ up], axis=1), convex))
    return polys


@dataclass
class Frame:
    """Orthographic screen (u, v) -> canvas pixels:  x = cx + (u - u0) * s ;  y = cy - (v - v0) * s."""
    s: float
    u0: float
    v0: float
    cx: float
    cy: float
    S: int

    def to_px(self, uv: np.ndarray) -> np.ndarray:
        out = np.empty_like(uv, dtype=float)
        out[:, 0] = self.cx + (uv[:, 0] - self.u0) * self.s
        out[:, 1] = self.cy - (uv[:, 1] - self.v0) * self.s
        return out

    def to_uv(self, x: float, y: float) -> tuple:
        return ((x - self.cx) / self.s + self.u0, -(y - self.cy) / self.s + self.v0)


def make_frame(polys, target: Target) -> Optional[Frame]:
    allp = np.vstack([p for p, _ in polys])
    lo, hi = allp.min(0), allp.max(0)
    wp, hp = hi - lo
    if wp < 1e-6 or hp < 1e-6:
        return None
    # similarity (uniform scale + centring) that best matches the target extents
    s = (target.W * wp + target.H * hp) / (wp * wp + hp * hp)
    c = (lo + hi) / 2
    return Frame(float(s), float(c[0]), float(c[1]), target.cx, target.cy, target.S)


def render(asm: Assembly, cam: Camera, target: Target, ov=None) -> Optional[np.ndarray]:
    polys = _project(resolve(asm, ov), cam)
    fr = make_frame(polys, target)
    if fr is None:
        return None
    img = np.zeros((target.S, target.S), np.uint8)
    for uv, convex in polys:
        pts = np.round(fr.to_px(uv)).astype(np.int32)
        if convex:
            cv2.fillConvexPoly(img, cv2.convexHull(pts), 1)
        else:
            cv2.fillPoly(img, [pts], 1)
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
    fixed = {"orthographic_front": (0, 0, 0), "orthographic_side": (90, 0, 0), "orthographic_top": (0, 89, 0)}
    if hint.projection in fixed:
        return Camera(*fixed[hint.projection])

    # Check target mask orientation for diagonal or turned objects
    rolls = [0.0]
    cnts, _ = cv2.findContours(target.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), _ = rect
            major, minor = max(rw, rh), min(rw, rh)
            if major / max(minor, 1.0) >= 1.4:
                pts = cnt.reshape(-1, 2).astype(np.float32)
                mean, eigenvectors = cv2.PCACompute(pts, mean=np.empty((0)))
                rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
                deg = np.degrees(rad)
                # Test in-plane rotation candidates
                rolls = [deg - 90, deg + 90, deg, 0.0]

    if hint.azimuth_deg is not None and hint.elevation_deg is not None:
        azs = [hint.azimuth_deg + d for d in (-20, -10, 0, 10, 20)]
        els = [max(2, hint.elevation_deg + d) for d in (-15, 0, 15)]
    else:
        azs = list(range(-165, 180, 20))
        els = [15, 30, 45, 60]

    best, best_s = Camera(35, 25, 0.0), -1.0
    for roll in rolls:
        for az in azs:
            for el in els:
                c = Camera(az, el, roll)
                img = render(asm, c, target)
                if img is None:
                    continue
                iou, ch = score(img, target)
                if iou - ch > best_s:
                    best, best_s = c, iou - ch
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
    part_ious: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


def per_part_iou(asm: Assembly, cam: Camera, target: Target, ov=None) -> dict:
    """Project each solid individually and compute the fraction of its projected
    silhouette that lands inside the target sketch mask. Catches floating / protruding parts."""
    solids = resolve(asm, ov)
    fr = make_frame(_project(solids, cam), target)
    if fr is None:
        return {p.id: 0.0 for p in asm.parts}

    right, up = basis(cam.az, cam.el, getattr(cam, "roll", 0.0))
    res = {}
    for pid, s in solids.items():
        if s.op != "union":
            continue
        p_img = np.zeros((target.S, target.S), np.uint8)
        for p3, convex in silhouette_polys(s):
            uv = np.stack([p3 @ right, p3 @ up], axis=1)
            pts = np.round(fr.to_px(uv)).astype(np.int32)
            if convex:
                cv2.fillConvexPoly(p_img, cv2.convexHull(pts), 1)
            else:
                cv2.fillPoly(p_img, [pts], 1)
        part_area = float(p_img.sum())
        if part_area == 0:
            res[pid] = 0.0
        else:
            in_target = float((p_img & target.mask).sum())
            res[pid] = round(in_target / part_area, 3)
    return res


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
    has_roll = getattr(cam, "roll", 0.0) != 0.0
    if fit_camera:
        bounds += [(cam.az - 20, cam.az + 20), (max(2.0, cam.el - 15), min(85.0, cam.el + 15))]
        x0 += [cam.az, cam.el]
        if has_roll:
            bounds.append((cam.roll - 25, cam.roll + 25))
            x0.append(cam.roll)
    x0 = np.array(x0)
    log0 = x0[:n].copy()
    counter = {"n": 0}

    def unpack(x):
        ov = {(pid, f): float(np.exp(x[i])) for i, (pid, f, _) in enumerate(slots)}
        if fit_camera:
            if has_roll:
                c = Camera(float(x[n]), float(x[n + 1]), float(x[n + 2]))
            else:
                c = Camera(float(x[n]), float(x[n + 1]), 0.0)
        else:
            c = cam
        return ov, c

    history = []
    best_seen = {"iou": 0.0}

    def base_loss(x):
        counter["n"] += 1
        ov, c = unpack(x)
        img = render(asm, c, target, ov)
        if img is None:
            return 10.0, None
        iou, ch = score(img, target)
        if iou > best_seen["iou"] + 0.015:
            best_seen["iou"] = iou
            history.append({"iter": counter["n"], "iou": round(float(iou), 3)})
        return (1.0 - iou) + 2.0 * ch, img

    def objective(x):
        l, _ = base_loss(x)
        return l + lam * float(np.mean((x[:n] - log0) ** 2))

    _, img0 = base_loss(x0)
    init_iou = score(img0, target)[0] if img0 is not None else 0.0
    if not history:
        history.append({"iter": 0, "iou": round(float(init_iou), 3)})

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
    history.append({"iter": len(history), "iou": round(float(iou), 3)})

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
    part_ious = per_part_iou(asm, c, target, ov)

    notes = []
    if iou < 0.8:
        notes.append("low silhouette agreement: structure is probably wrong, not just the dims")

    # Check for floating/protruding parts
    for pid, p_iou in part_ious.items():
        if p_iou < 0.5:
            notes.append(f"Part '{pid}' precision low ({int(p_iou*100)}% in silhouette) - possible protruding/floating geometry")

    return FitResult(iou, init_iou, c, ov, sens_slot, sens_part, img, counter["n"], notes, part_ious, history)


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


def silhouette_iou(asm_or_img, cam_or_target, target: Optional[Target] = None, ov=None) -> float:
    """
    Computes real silhouette IoU between an assembly render and the target mask.
    Accepts:
      silhouette_iou(asm, cam, target)
      silhouette_iou(asm, target, cam)
      silhouette_iou(img, target)
    """
    if isinstance(cam_or_target, Target):
        tgt = cam_or_target
        if isinstance(asm_or_img, np.ndarray):
            return float(score(asm_or_img, tgt)[0])
        cam = target if isinstance(target, Camera) else Camera(35, 25)
    else:
        cam = cam_or_target
        tgt = target

    img = render(asm_or_img, cam, tgt, ov) if not isinstance(asm_or_img, np.ndarray) else asm_or_img
    if img is None or tgt is None:
        return 0.0
    return float(score(img, tgt)[0])

