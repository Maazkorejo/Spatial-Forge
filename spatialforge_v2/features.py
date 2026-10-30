"""
Feature layer: buttons, dials, screws, holes, slots, switches.

Placement is MEASURED, not generated:
    detection box (original px) -> canvas px -> screen (u, v) -> orthographic ray -> first hit
    on the fitted body -> (part, face, normalised u/v) + foreshortening-corrected size.
`verified` only means: the ray hit a visible surface at a non-grazing angle and the detector was
confident. It does NOT mean the real size or depth is right (depth is always `assumed`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fit import Camera, Frame, Target, basis
from geometry import feature_mesh_local, part_mesh, transform
from raster import view_dir
from schema import Assembly, Dim, Feature, FeatureSite, ROUND_FEATURES
from solver import FACE, resolve

TANG = {0: (1, 2), 1: (0, 2), 2: (0, 1)}                     # tangent axes (t1, t2) per normal axis
FACE_OF = {(ax, sg): name for name, (ax, sg) in FACE.items()}

KEYWORDS = [("screw", ("screw", "bolt", "rivet")), ("dial", ("dial", "knob")),
            ("switch", ("switch", "toggle", "rocker")), ("button", ("button", "btn", "key")),
            ("slot", ("slot", "opening", "cutout", "vent")), ("hole", ("hole", "port", "socket", "jack"))]


@dataclass
class Detection:
    label: str
    box: tuple                 # (x0, y0, x1, y1) in ORIGINAL image pixels
    score: float = 1.0


def label_to_kind(label: str) -> Optional[str]:
    l = label.lower()
    for kind, words in KEYWORDS:
        if any(w in l for w in words):
            return kind
    return None


# ----------------------------------------------------------------------------- ray casting
def ray_mesh(o: np.ndarray, d: np.ndarray, V: np.ndarray, F: np.ndarray):
    """Moller-Trumbore, two-sided. Returns (t, tri_index) of the nearest hit with t > 0, or None."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = o - p0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    if not hit.any():
        return None
    i = int(np.argmin(np.where(hit, t, np.inf)))
    return float(t[i]), i


def cast(o, d, meshes: dict):
    """Nearest hit over several meshes -> (name, t, point, normal facing the ray origin) or None."""
    best = None
    for name, (V, F) in meshes.items():
        r = ray_mesh(o, d, V, F)
        if r and (best is None or r[0] < best[1]):
            best = (name, r[0], r[1])
    if best is None:
        return None
    name, t, i = best
    V, F = meshes[name]
    n = np.cross(V[F[i, 1]] - V[F[i, 0]], V[F[i, 2]] - V[F[i, 0]])
    n /= np.linalg.norm(n)
    if n @ d > 0:
        n = -n
    return name, t, o + t * d, n


# ----------------------------------------------------------------------------- placement
def _dim(v: float, src: str, lo=0.6, hi=1.5) -> Dim:
    return Dim(value=float(v), min=float(v * lo), max=float(v * hi), source=src)


def _round_size(w_u: float, h_u: float, n, right, up) -> float:
    """Disc of diameter D with normal n: projected extents are D*sqrt(1-(n.r)^2) and D*sqrt(1-(n.u)^2)."""
    kw = np.sqrt(max(1 - (n @ right) ** 2, 0.0))
    kh = np.sqrt(max(1 - (n @ up) ** 2, 0.0))
    est = []
    if kw > 0.25:
        est.append(w_u / kw)
    if kh > 0.25:
        est.append(h_u / kh)
    return float(np.mean(est)) if est else float(max(w_u, h_u))


def _rect_size(w_u, h_u, e1, e2, right, up):
    """Projected bbox of an a x b rectangle spanned by e1, e2:  w = a|e1.r| + b|e2.r|,  h = a|e1.u| + b|e2.u|."""
    M = np.array([[abs(e1 @ right), abs(e2 @ right)], [abs(e1 @ up), abs(e2 @ up)]])
    if abs(np.linalg.det(M)) > 0.1 and np.linalg.cond(M) < 6:
        a, b = np.linalg.solve(M, [w_u, h_u])
        if a > 0 and b > 0:
            return float(a), float(b)
    return float(w_u), float(h_u)               # ill-conditioned: treat as frontal


def _default_depth(kind: str, dims: dict) -> float:
    m = dims.get("diameter") or min(dims["size_u"], dims["size_v"])
    return {"button": 0.30, "dial": 0.50, "screw": 0.15, "hole": 0.8, "slot": 0.5, "switch": 0.3}[kind] * m


def place_features(asm: Assembly, cam: Camera, frame: Frame, target: Target, detections: list,
                   ov=None, min_cos: float = 0.15, min_score: float = 0.2):
    """Returns (features, rejected) where rejected = [(label, reason)]."""
    solids = resolve(asm, ov)
    meshes = {pid: part_mesh(s) for pid, s in solids.items() if s.op == "union"}
    right, up = basis(cam.az, cam.el)
    d = view_dir(cam)
    far = 10 * max(float(np.max(s.size)) for s in solids.values()) + 100
    feats, rejected, counts = [], [], {}

    for det in detections:
        kind = label_to_kind(det.label)
        if kind is None:
            rejected.append((det.label, "unknown feature type"))
            continue
        if det.score < min_score:
            rejected.append((det.label, f"detector score {det.score:.2f} < {min_score}"))
            continue
        x0, y0, x1, y1 = det.box
        cx, cy = target.to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
        u, v = frame.to_uv(cx, cy)
        hit = cast(u * right + v * up + far * d, -d, meshes)
        if hit is None:
            rejected.append((det.label, "ray missed the fitted body"))
            continue
        pid, _, P, n = hit
        cosinc = float(abs(n @ d))
        if cosinc < min_cos:
            rejected.append((det.label, f"grazing view (cos={cosinc:.2f})"))
            continue

        ax = int(np.argmax(np.abs(n)))
        sg = 1 if n[ax] > 0 else -1
        t1, t2 = TANG[ax]
        s = solids[pid]
        lo = s.center - s.size / 2
        uu = float(np.clip((P[t1] - lo[t1]) / s.size[t1], 0, 1))
        vv = float(np.clip((P[t2] - lo[t2]) / s.size[t2], 0, 1))

        w_u = (x1 - x0) * target.kx / frame.s
        h_u = (y1 - y0) * target.ky / frame.s
        src = "inferred" if cosinc >= 0.5 else "assumed"
        e1, e2 = np.eye(3)[t1], np.eye(3)[t2]
        counts[kind] = counts.get(kind, 0) + 1
        fid = f"{kind}_{counts[kind]}"
        kw = dict(id=fid, kind=kind, on=FeatureSite(part=pid, face=FACE_OF[(ax, sg)], u=uu, v=vv),
                  layer="cosmetic" if kind == "screw" else "feature",
                  detector_score=float(det.score), incidence_cos=cosinc,
                  verified=bool(cosinc >= 0.35 and det.score >= 0.3))
        if kind in ROUND_FEATURES:
            D = np.clip(_round_size(w_u, h_u, n, right, up), 0.5, 0.6 * min(s.size[t1], s.size[t2]))
            dims = {"diameter": D}
            kw["diameter"] = _dim(D, src)
        else:
            a, b = _rect_size(w_u, h_u, e1, e2, right, up)
            a = float(np.clip(a, 0.5, 0.8 * s.size[t1]))
            b = float(np.clip(b, 0.5, 0.8 * s.size[t2]))
            dims = {"size_u": a, "size_v": b}
            kw["size_u"], kw["size_v"] = _dim(a, src), _dim(b, src)
        kw["depth"] = _dim(_default_depth(kind, dims), "assumed", 0.4, 2.0)
        feats.append(Feature(**kw))
    return feats, rejected


# ----------------------------------------------------------------------------- realisation
def feature_pose(f: Feature, solids: dict, meshes: dict):
    """(origin, R) with R columns (e1, e2, normal). Re-ray-casts from the part's AABB face so the
    feature follows the real surface (and any later dimension change)."""
    s = solids[f.on.part]
    ax, sg = FACE[f.on.face]
    t1, t2 = TANG[ax]
    lo = s.center - s.size / 2
    P = s.center.copy()
    P[ax] += sg * s.size[ax] / 2
    P[t1] = lo[t1] + f.on.u * s.size[t1]
    P[t2] = lo[t2] + f.on.v * s.size[t2]
    n0 = np.zeros(3)
    n0[ax] = sg
    hit = cast(P + n0 * (s.size[ax] + 1.0), -n0, {f.on.part: meshes[f.on.part]})
    origin, n = (hit[2], hit[3]) if hit else (P, n0)
    e1 = np.eye(3)[t1] - n * (np.eye(3)[t1] @ n)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.eye(3)[t2] - n * (np.eye(3)[t2] @ n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return origin, np.column_stack([e1, e2, n])


def feature_world_meshes(asm: Assembly, ov=None) -> list:
    """[{id, kind, layer, part, mesh, cutter}] in world coordinates (u units)."""
    solids = resolve(asm, ov)
    meshes = {pid: part_mesh(s) for pid, s in solids.items() if s.op == "union"}
    out = []
    for f in asm.features:
        origin, R = feature_pose(f, solids, meshes)
        dims = {k: d.value for k, d in f.dims().items()}
        Vc, Fc = feature_mesh_local(f.kind, dims, cutter=True)
        Vp, Fp = feature_mesh_local(f.kind, dims, cutter=False)
        out.append({"id": f.id, "kind": f.kind, "layer": f.layer, "part": f.on.part,
                    "is_cut": f.kind in ("hole", "slot"),
                    "mesh": (transform(Vp, R, origin), Fp), "cutter": (transform(Vc, R, origin), Fc),
                    "origin": origin, "normal": R[:, 2], "R": R})
    return out
