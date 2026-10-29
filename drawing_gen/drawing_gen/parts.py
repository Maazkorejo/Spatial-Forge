"""
Random ground-truth parts. Two families, matching the scope we agreed to support first:
  * prismatic 2.5D: block (+ optional L-notch / top step / top chamfer) with vertical through holes, a counterbore,
    a blind top pocket, a bottom pocket (visible ONLY as hidden lines), and a horizontal through hole.
  * turned: stepped shaft along x with end chamfers, a cross hole and an optional blind centre bore.
Each part returns (cq.Workplane, spec). `spec["features"]` lists what was cut, and `spec["hull_exact"]`
is False when three silhouettes cannot determine the solid: blind pockets / counterbores / bores (hidden-line-only
features) AND every turned part (round cross-section).
Dimensions are whole millimetres so tests can assert exact values.
"""
from __future__ import annotations

import cadquery as cq
import numpy as np


def _cyl_z(d, h, x, y, z0=0.0):
    return cq.Workplane("XY").circle(d / 2).extrude(h).translate((x, y, z0))


def prismatic(rng: np.random.Generator):
    W = int(rng.choice(range(40, 121, 5)))
    D = int(rng.choice(range(30, 81, 5)))
    H = int(rng.choice(range(15, 41, 5)))
    body = cq.Workplane("XY").box(W, D, H, centered=False)
    feats, hull_exact = [], True

    if rng.random() < 0.4:                                   # L-notch through full height
        nx, ny = int(W * rng.choice([0.25, 0.3, 0.4]) // 5 * 5), int(D * rng.choice([0.25, 0.3, 0.4]) // 5 * 5)
        if nx >= 10 and ny >= 10:
            body = body.cut(cq.Workplane("XY").box(nx, ny, H, centered=False).translate((W - nx, D - ny, 0)))
            feats.append({"type": "notch", "nx": nx, "ny": ny})
    notch = next((f for f in feats if f["type"] == "notch"), None)

    if rng.random() < 0.35:                                  # step on the top, full depth
        sx, sz = int(rng.choice([10, 15, 20])), int(rng.choice([5, 10]))
        if sz < H - 5 and sx < W - 20:
            body = body.cut(cq.Workplane("XY").box(sx, D, sz, centered=False).translate((0, 0, H - sz)))
            feats.append({"type": "step", "sx": sx, "sz": sz})
    if rng.random() < 0.3:
        c = float(rng.choice([1, 2]))
        body = body.faces(">Z").edges().chamfer(c)
        feats.append({"type": "chamfer", "c": c})

    def free(x, y, r):                                       # inside footprint, outside the notch, margin from edges
        m = r + 4
        if not (m <= x <= W - m and m <= y <= D - m):
            return False
        return not (notch and x + m > W - notch["nx"] and y + m > D - notch["ny"])

    placed = []
    n_holes = int(rng.integers(1, 4))
    for _ in range(n_holes * 8):
        if len(placed) >= n_holes:
            break
        d = float(rng.choice([4, 5, 6, 8, 10]))
        x, y = float(rng.integers(8, W - 8)), float(rng.integers(8, D - 8))
        if free(x, y, d) and all(np.hypot(x - px, y - py) > (d + pd) / 2 + 8 for px, py, pd in placed):
            placed.append((x, y, d))
            body = body.cut(_cyl_z(d, H + 2, x, y, -1))
            feats.append({"type": "hole_z", "x": x, "y": y, "d": d})
    if rng.random() < 0.3:                                   # obround through slot along z (gives ARCS in the top view)
        for _ in range(20):
            sl, sw = float(rng.choice([14, 18, 22])), float(rng.choice([6, 8]))
            x, y = float(rng.integers(int(sl / 2 + 6), max(int(sl / 2 + 7), int(W - sl / 2 - 5)))), float(rng.integers(int(sw / 2 + 6), max(int(sw / 2 + 7), int(D - sw / 2 - 5))))
            if free(x, y, sl / 2) and all(np.hypot(x - px, y - py) > sl / 2 + pd / 2 + 6 for px, py, pd in placed):
                body = body.cut(cq.Workplane("XY").slot2D(sl, sw).extrude(H + 2).translate((x, y, -1)))
                feats.append({"type": "slot_z", "x": x, "y": y, "length": sl, "width": sw})
                placed.append((x, y, sl))
                break
    if placed and rng.random() < 0.4:                        # counterbore on the first hole (top side)
        x, y, d = placed[0]
        cb_d, cb_h = d * 1.8, max(3.0, H * 0.35)
        if free(x, y, cb_d / 2 + 2):
            body = body.cut(_cyl_z(cb_d, cb_h, x, y, H - cb_h))
            feats.append({"type": "counterbore", "x": x, "y": y, "d": cb_d, "depth": cb_h})
            hull_exact = False
    if rng.random() < 0.5:                                   # blind rectangular pocket from the top
        for _ in range(20):
            sx, sy = float(rng.choice([10, 15, 20])), float(rng.choice([8, 10, 15]))
            x, y = float(rng.integers(6, max(7, int(W - sx - 5)))), float(rng.integers(6, max(7, int(D - sy - 5))))
            cx, cy = x + sx / 2, y + sy / 2
            if free(cx, cy, max(sx, sy) / 2) and all(np.hypot(cx - px, cy - py) > max(sx, sy) / 2 + pd / 2 + 4 for px, py, pd in placed):
                depth = float(round(H * rng.choice([0.3, 0.5])))
                body = body.cut(cq.Workplane("XY").box(sx, sy, depth, centered=False).translate((x, y, H - depth)))
                feats.append({"type": "pocket_top", "x": x, "y": y, "sx": sx, "sy": sy, "depth": depth})
                placed.append((cx, cy, max(sx, sy)))
                hull_exact = False
                break
    if rng.random() < 0.25:                                  # pocket from the bottom: ONLY hidden lines in the views
        for _ in range(20):
            sx, sy = float(rng.choice([10, 15])), float(rng.choice([8, 10]))
            x, y = float(rng.integers(6, max(7, int(W - sx - 5)))), float(rng.integers(6, max(7, int(D - sy - 5))))
            cx, cy = x + sx / 2, y + sy / 2
            if free(cx, cy, max(sx, sy) / 2) and all(np.hypot(cx - px, cy - py) > max(sx, sy) / 2 + pd / 2 + 4 for px, py, pd in placed):
                depth = float(round(H * 0.3))
                body = body.cut(cq.Workplane("XY").box(sx, sy, depth, centered=False).translate((x, y, 0)))
                feats.append({"type": "pocket_bottom", "x": x, "y": y, "sx": sx, "sy": sy, "depth": depth, "hidden_only": True})
                hull_exact = False
                break
    if H >= 20 and rng.random() < 0.3:                       # horizontal through hole along y
        d = float(rng.choice([5, 6, 8]))
        x, z = float(rng.integers(10, W - 10)), float(H / 2)
        if not any(f["type"] in ("pocket_top", "counterbore") for f in feats):
            body = body.cut(cq.Workplane("XZ").circle(d / 2).extrude(-(D + 2)).translate((x, -1, z)))
            feats.append({"type": "hole_y", "x": x, "z": z, "d": d})
    return body, {"family": "prismatic", "W": W, "D": D, "H": H, "features": feats, "hull_exact": hull_exact}


def turned(rng: np.random.Generator):
    n = int(rng.integers(2, 5))
    radii = [float(rng.integers(6, 26))]
    while len(radii) < n:                                    # adjacent radii must differ by >= 2 (no zero-length edges)
        r = float(rng.integers(6, 26))
        if abs(r - radii[-1]) >= 2:
            radii.append(r)
    lens = [float(rng.choice([10, 15, 20, 25, 30, 40])) for _ in range(n)]
    ch = float(rng.choice([1, 2]))
    pts = [(0.0, 0.0), (0.0, radii[0] - ch), (ch, radii[0])]
    x = 0.0
    for i in range(n):
        x_end = x + lens[i]
        if i > 0:
            pts.append((x, radii[i]))                        # shoulder
        if i < n - 1:
            pts.append((x_end, radii[i]))
        else:
            pts += [(x_end - ch, radii[i]), (x_end, radii[i] - ch), (x_end, 0.0)]
        x = x_end
    prof = cq.Workplane("XY").polyline(pts).close()
    body = prof.revolve(360, (0, 0, 0), (1, 0, 0))
    L = x
    feats, hull_exact = [], True
    big = int(np.argmax(radii))
    x0 = sum(lens[:big])
    if rng.random() < 0.5 and lens[big] >= 15:               # cross hole along z through the biggest section
        d = float(rng.choice([4, 5, 6, 8]))
        d = min(d, radii[big])
        cx = x0 + lens[big] / 2
        body = body.cut(_cyl_z(d, 2 * radii[big] + 4, cx, 0, -radii[big] - 2))
        feats.append({"type": "cross_hole", "x": cx, "d": d})
    if rng.random() < 0.4:                                   # blind centre bore from x = 0
        d, depth = float(round(radii[0] * 0.8)), float(round(min(lens[0] * 0.7, L * 0.5)))
        if d >= 4 and depth >= 5:
            body = body.cut(cq.Workplane("YZ").circle(d / 2).extrude(depth))
            feats.append({"type": "bore", "d": d, "depth": depth})
            hull_exact = False
    # A body of revolution is NEVER recovered exactly by Front ∩ Top ∩ Side: the hull has a rounded-square cross
    # section (measured in test_drawings.py). Silhouettes alone cannot give a round section -> revolve detection needed.
    return body, {"family": "turned", "L": L, "radii": radii, "lens": lens, "chamfer": ch,
                  "features": feats, "hull_exact": False, "hull_inexact_reason": "round cross-section"}


def make_part(rng: np.random.Generator, family: str | None = None):
    family = family or ("prismatic" if rng.random() < 0.7 else "turned")
    body, spec = prismatic(rng) if family == "prismatic" else turned(rng)
    shape = body.val()
    spec["volume"] = float(shape.Volume())
    spec["valid"] = bool(shape.isValid())
    return body, spec
