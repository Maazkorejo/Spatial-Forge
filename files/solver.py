"""
Relation solver: turns the relational DSL into world-space axis-aligned solids.
Deterministic, no optimisation. Because every part is placed relative to its parent's
current size, changing any dimension keeps the assembly connected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from schema import Assembly, Part

# face -> (axis index, sign)
FACE = {"right": (0, 1), "left": (0, -1), "back": (1, 1), "front": (1, -1),
        "top": (2, 1), "bottom": (2, -1)}
AX = {"x": 0, "y": 1, "z": 2}

Overrides = Optional[dict]  # {(part_id, field): value}


@dataclass
class Solid:
    id: str
    kind: str
    op: str
    cuts: list
    center: np.ndarray
    size: np.ndarray          # full AABB extents
    axis: Optional[str]

    def aabb(self):
        h = self.size / 2
        return self.center - h, self.center + h


def _val(p: Part, f: str, ov: Overrides) -> float:
    if ov is not None and (p.id, f) in ov:
        return ov[(p.id, f)]
    return getattr(p, f).value


def _size(p: Part, ov: Overrides) -> np.ndarray:
    if p.kind == "box":
        return np.array([_val(p, "width_x", ov), _val(p, "depth_y", ov), _val(p, "height_z", ov)])
    d = _val(p, "diameter", ov)
    if p.kind == "sphere":
        return np.array([d, d, d])
    L = _val(p, "length", ov)
    return {"x": np.array([L, d, d]), "y": np.array([d, L, d]), "z": np.array([d, d, L])}[p.axis]


def resolve(asm: Assembly, ov: Overrides = None) -> dict[str, Solid]:
    solids: dict[str, Solid] = {}
    for p in asm.topo():
        size = _size(p, ov)
        c = np.zeros(3)
        pl = p.placement
        if pl is None:
            c[:] = (0.0, 0.0, size[2] / 2)
        else:
            par = solids[pl.parent]
            n, s = FACE[pl.face]
            aligns = (pl.align_x, pl.align_y, pl.align_z)
            offs = (pl.offset_x, pl.offset_y, pl.offset_z)
            for a in range(3):
                if a == n:
                    sink = pl.embed * size[a]
                    c[a] = par.center[a] + s * (par.size[a] / 2 + size[a] / 2 - sink)
                else:
                    if aligns[a] == "center":
                        c[a] = par.center[a]
                    elif aligns[a] == "min":
                        c[a] = par.center[a] - par.size[a] / 2 + size[a] / 2
                    else:
                        c[a] = par.center[a] + par.size[a] / 2 - size[a] / 2
                    c[a] += offs[a]
        solids[p.id] = Solid(p.id, p.kind, p.op, list(p.cuts), c, size, p.axis)
    return solids


def surface_points(s: Solid, n: int = 24) -> np.ndarray:
    """Points whose projected convex hull equals (or closely approximates) the silhouette."""
    h = s.size / 2
    if s.kind == "box":
        sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return s.center + sg * h
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if s.kind == "cylinder":
        a = AX[s.axis]
        b, c2 = [i for i in range(3) if i != a]
        pts = []
        for end in (-1, 1):
            p = np.zeros((n, 3))
            p[:, a] = end * h[a]
            p[:, b] = h[b] * np.cos(t)
            p[:, c2] = h[c2] * np.sin(t)
            pts.append(p)
        return s.center + np.vstack(pts)
    pts = []  # sphere: three great circles
    for a, b in ((0, 1), (0, 2), (1, 2)):
        p = np.zeros((n, 3))
        p[:, a] = h[a] * np.cos(t)
        p[:, b] = h[b] * np.sin(t)
        pts.append(p)
    return s.center + np.vstack(pts)


def _overlap(a: Solid, b: Solid, tol=1e-6) -> bool:
    (al, ah), (bl, bh) = a.aabb(), b.aabb()
    return bool(np.all(ah > bl + tol) and np.all(bh > al + tol))


def geometry_report(asm: Assembly, ov: Overrides = None) -> dict:
    """Cheap validity checks. `errors` => reject candidate, `warnings` => penalise."""
    solids = resolve(asm, ov)
    errors, warnings = [], []
    pos = [s for s in solids.values() if s.op == "union"]
    lo = np.min([s.aabb()[0] for s in pos], axis=0)
    hi = np.max([s.aabb()[1] for s in pos], axis=0)
    ext = hi - lo
    longest = float(ext.max())

    if ext.min() / longest < 0.02:
        errors.append("Overall bounding box is degenerate (one extent < 2% of the longest).")
    if ext.max() / max(ext.min(), 1e-9) > 40:
        warnings.append("Extreme overall aspect ratio (>40:1).")
    if not (30 <= longest <= 300):
        warnings.append(f"Longest extent is {longest:.0f}u; the prompt asks for ~100u.")

    for s in solids.values():
        if s.size.min() / longest < 0.01:
            warnings.append(f"Part '{s.id}' has a dimension < 1% of the object (tiny detail?).")

    for s in solids.values():
        if s.op == "difference":
            for t in s.cuts:
                if not _overlap(s, solids[t]):
                    errors.append(f"Cut '{s.id}' does not intersect its target '{t}'.")

    for a in pos:                       # fully buried parts add nothing
        for b in pos:
            if a is not b:
                (al, ah), (bl, bh) = a.aabb(), b.aabb()
                if np.all(al >= bl - 1e-6) and np.all(ah <= bh + 1e-6):
                    warnings.append(f"Part '{a.id}' is entirely inside '{b.id}'.")
                    break

    return {"errors": errors, "warnings": warnings,
            "extent": ext.tolist(), "longest": longest, "n_parts": len(asm.parts)}
