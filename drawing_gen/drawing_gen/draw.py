"""
Drawing layout: three views + centerlines + dimensions + title block, as a list of entities (IR).

Entity: {"kind": line|circle|arc|text|dim_linear|dim_diam, "g": {...}, "layer", "ltype", "role"}
  `role` is ground-truth metadata (visible|hidden|center|dim|title) and is NOT written to the DXF.
Placement (images are identical in both standards; only positions differ):
  third angle: top above front, right view to the RIGHT of front.
  first angle: top BELOW front, right view to the LEFT of front.
Scale s (1:1, 1:2, 2:1) multiplies drawn geometry. Dimension values stay REAL (what a drafter writes), so
drawn length / dimension value = s. That ratio is what the dimension cross-check must recover.
"""
from __future__ import annotations

import copy

import numpy as np

from project import VIEWS, bbox, project_view

GAP = 25.0
TEXT_H = 3.5
SCALES = {"1:1": 1.0, "1:2": 0.5, "2:1": 2.0}


def _ent(kind, g, role, layer=None, ltype=None):
    d = {"visible": ("VISIBLE", "CONTINUOUS"), "hidden": ("HIDDEN", "DASHED"), "center": ("CENTER", "CENTER"),
         "dim": ("DIM", "CONTINUOUS"), "title": ("TITLE", "CONTINUOUS")}[role]
    return {"kind": kind, "g": g, "layer": layer or d[0], "ltype": ltype or d[1], "role": role}


def _fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _prim_to_ents(p, tf, s, role):
    if p["t"] == "line":
        x0, y0, x1, y1 = p["p"]
        return [_ent("line", {"p": [*tf(x0, y0), *tf(x1, y1)]}, role)]
    if p["t"] == "circle":
        return [_ent("circle", {"c": list(tf(*p["c"])), "r": p["r"] * s}, role)]
    if p["t"] == "arc":
        return [_ent("arc", {"c": list(tf(*p["c"])), "r": p["r"] * s, "a0": p["a0"], "a1": p["a1"]}, role)]
    pts = [tf(*q) for q in p["pts"]]
    return [_ent("line", {"p": [*pts[i], *pts[i + 1]]}, role) for i in range(len(pts) - 1)]


def build_drawing(solid, spec: dict, standard: str = "third", scale: str = "1:1", name: str = "PART-001") -> dict:
    s = SCALES[scale]
    views = {v: project_view(solid.val(), v) for v in VIEWS}
    box = {v: bbox(views[v]["visible"]) for v in views}
    wh = {v: (box[v][2] - box[v][0], box[v][3] - box[v][1]) for v in views}
    fw, fh = wh["front"]
    if standard == "third":
        origin = {"front": (0.0, 0.0), "top": (0.0, fh * s + GAP), "right": (fw * s + GAP, 0.0)}
    else:
        origin = {"front": (0.0, 0.0), "top": (0.0, -(wh["top"][1] * s + GAP)), "right": (-(wh["right"][0] * s + GAP), 0.0)}

    ents, dims = [], []
    tfs = {}
    for v in views:
        bx0, by0 = box[v][0], box[v][1]
        ox, oy = origin[v]
        tfs[v] = lambda x, y, bx0=bx0, by0=by0, ox=ox, oy=oy: ((x - bx0) * s + ox, (y - by0) * s + oy)
        for role in ("visible", "hidden"):
            for p in views[v][role]:
                for e in _prim_to_ents(p, tfs[v], s, role):
                    e["view"] = v
                    ents.append(e)
        for p in views[v]["visible"]:                        # centre marks on circles
            if p["t"] == "circle":
                cx, cy = tfs[v](*p["c"])
                r = p["r"] * s + 2.5
                for g in ([cx - r, cy, cx + r, cy], [cx, cy - r, cx, cy + r]):
                    e = _ent("line", {"p": g}, "center")
                    e["view"] = v
                    ents.append(e)
    if spec["family"] == "turned":                           # shaft axis in front and top views
        for v in ("front", "top"):
            ax0, ay = tfs[v](box[v][0] - 3 / s, 0.0)
            ax1, _ = tfs[v](box[v][2] + 3 / s, 0.0)
            e = _ent("line", {"p": [ax0, ay, ax1, ay]}, "center")
            e["view"] = v
            ents.append(e)

    # --- dimensions (real values) -------------------------------------------------------------
    def lin(view, orient, side):
        x0, y0, x1, y1 = box[view]
        ox, oy = origin[view]
        sx0, sy0, sx1, sy1 = tfs[view](x0, y0) + tfs[view](x1, y1)
        if orient == "h":
            p1, p2, base, val = (sx0, sy0), (sx1, sy0), (sx0, sy0 - 12), x1 - x0
        else:
            xx = sx0 - 14 if side == "left" else sx1 + 14
            p1, p2, base, val = (sx0 if side == "left" else sx1, sy0), (sx0 if side == "left" else sx1, sy1), (xx, sy0), y1 - y0
        e = _ent("dim_linear", {"p1": list(p1), "p2": list(p2), "base": list(base), "angle": 0 if orient == "h" else 90,
                                "text": "<>" if s == 1.0 else _fmt(val), "value_real": float(val)}, "dim")
        e["view"] = view
        ents.append(e)
        dims.append({"view": view, "orient": orient, "value_real": float(val), "drawn": float(val * s),
                     "text_override": s != 1.0, "kind": "linear"})

    lin("front", "h", "left")
    lin("front", "v", "right" if standard == "third" else "left")
    lin("top", "v", "left")
    ndia = 0
    for v in ("top", "right", "front"):
        for p in views[v]["visible"]:
            if p["t"] == "circle" and ndia < 2:
                c = tfs[v](*p["c"])
                e = _ent("dim_diam", {"center": list(c), "radius": p["r"] * s, "angle": 45,
                                      "text": "<>" if s == 1.0 else "Ø" + _fmt(2 * p["r"]), "value_real": float(2 * p["r"])}, "dim")
                e["view"] = v
                ents.append(e)
                dims.append({"view": v, "kind": "diameter", "value_real": float(2 * p["r"]), "drawn": float(2 * p["r"] * s),
                             "text_override": s != 1.0})
                ndia += 1

    # --- title block ---------------------------------------------------------------------------
    allpts = np.array([[v for v in e["g"]["p"][:2]] for e in ents if e["kind"] == "line"] +
                      [[v for v in e["g"]["p"][2:]] for e in ents if e["kind"] == "line"])
    minx, miny, maxx = float(allpts[:, 0].min()), float(allpts[:, 1].min()), float(allpts[:, 0].max())
    tx, ty = maxx - 90, miny - 55
    for g in ([tx, ty, tx + 90, ty], [tx + 90, ty, tx + 90, ty + 28], [tx + 90, ty + 28, tx, ty + 28], [tx, ty + 28, tx, ty]):
        ents.append(_ent("line", {"p": g}, "title"))
    lines = [f"TITLE: {name}", f"SCALE {scale}", "UNITS: MM", f"{standard.upper()} ANGLE PROJECTION"]
    for i, t in enumerate(lines):
        e = _ent("text", {"text": t, "at": [tx + 3, ty + 21 - i * 6], "height": TEXT_H}, "title")
        e["tag"] = ["title", "scale", "units", "projection"][i]
        ents.append(e)

    meta = {"scale": s, "scale_text": scale, "standard": standard, "units": "mm", "name": name,
            "view_origin": {k: list(v) for k, v in origin.items()},
            "view_box_model": {k: list(v) for k, v in box.items()}, "dims": dims}
    return {"ents": ents, "meta": meta}


def clone(d: dict) -> dict:
    return copy.deepcopy(d)
