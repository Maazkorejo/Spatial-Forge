"""
Dirty-file noise injection. Every defect is independent, seeded, and logged, so failures can be attributed.

  gaps         endpoints moved by up to `gap` mm (lines: positional; arcs: angular). Breaks closed loops.
  fragments    lines split into 2-4 pieces
  overlaps     lines extended past their ends; duplicated lines (tiny offset)
  explode_dashes  dashed hidden lines become many short CONTINUOUS segments (no linetype left to read)
  explode_dims    native DIMENSION -> raw lines (witness, dimension line, arrowheads) + TEXT
  collapse     every entity moved to layer "0"
  units/scale  $INSUNITS = 0; "SCALE x:y" title text removed
Ground-truth `role` stays on every entity so a classifier can be scored later.
"""
from __future__ import annotations

import copy
import math

import numpy as np

LEVELS = {
    "clean": {},
    "light": dict(gap=0.05, p_gap=0.5, frag=0.15, overlap=0.10, dup=0.0, p_units_drop=0.3, p_scale_drop=0.0,
                  collapse=False, explode_dashes=False, explode_dims=False),
    "heavy": dict(gap=0.5, p_gap=0.8, frag=0.35, overlap=0.30, dup=0.10, p_units_drop=0.7, p_scale_drop=0.5,
                  collapse=True, explode_dashes=True, explode_dims=True),
}


def _ln(p, role, view=None, layer="DIM", ltype="CONTINUOUS"):
    e = {"kind": "line", "g": {"p": [float(v) for v in p]}, "layer": layer, "ltype": ltype, "role": role}
    if view:
        e["view"] = view
    return e


def _arrow(tip, direction, view):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    out = []
    for sgn in (-1, 1):
        a = math.radians(15) * sgn
        r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ d
        out.append(_ln([*tip, *(np.asarray(tip) + r * 2.5)], "dim", view))
    return out


def _fmt(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def explode_dim(e: dict) -> list:
    g, view = e["g"], e.get("view")
    out = []
    if e["kind"] == "dim_linear":
        p1, p2, base = np.array(g["p1"]), np.array(g["p2"]), np.array(g["base"])
        ax = np.array([1.0, 0.0]) if g["angle"] == 0 else np.array([0.0, 1.0])
        pr = np.array([0.0, 1.0]) if g["angle"] == 0 else np.array([1.0, 0.0])
        off = float((base - p1) @ pr)
        q1, q2 = p1 + pr * off, p2 + pr * off
        sg = 1 if off >= 0 else -1
        out += [_ln([*(p1 + pr * sg * 1.0), *(q1 + pr * sg * 1.5)], "dim", view),
                _ln([*(p2 + pr * sg * 1.0), *(q2 + pr * sg * 1.5)], "dim", view),
                _ln([*q1, *q2], "dim", view)]
        out += _arrow(q1, q2 - q1, view) + _arrow(q2, q1 - q2, view)
        mid = (q1 + q2) / 2 + pr * 1.0
        t = {"kind": "text", "g": {"text": e["g"]["text"] if e["g"]["text"] != "<>" else _fmt(e["g"]["value_real"]),
                                   "at": [float(mid[0]), float(mid[1])], "height": 3.5}, "layer": "DIM", "ltype": "CONTINUOUS", "role": "dim"}
        out.append(t)
    else:
        c, r, a = np.array(g["center"]), g["radius"], math.radians(g["angle"])
        d = np.array([math.cos(a), math.sin(a)])
        tip = c + d * r
        end = c + d * (r + 12)
        out += [_ln([*tip, *end], "dim", view)] + _arrow(tip, -d, view)
        out.append({"kind": "text", "g": {"text": g["text"] if g["text"] != "<>" else "Ø" + _fmt(g["value_real"]),
                                          "at": [float(end[0]), float(end[1])], "height": 3.5},
                    "layer": "DIM", "ltype": "CONTINUOUS", "role": "dim"})
    for o in out:
        if view:
            o["view"] = view
    return out


def explode_dashes(e: dict, dash=3.0, space=1.5) -> list:
    g, out = e["g"], []

    def mk(kind, gg):
        n = {"kind": kind, "g": gg, "layer": e["layer"], "ltype": "CONTINUOUS", "role": "hidden"}
        if "view" in e:
            n["view"] = e["view"]
        return n

    if e["kind"] == "line":
        x0, y0, x1, y1 = g["p"]
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        t = 0.0
        while t < L - 1e-6:
            t2 = min(t + dash, L)
            out.append(mk("line", {"p": [x0 + ux * t, y0 + uy * t, x0 + ux * t2, y0 + uy * t2]}))
            t = t2 + space
        return out or [e]
    if e["kind"] in ("circle", "arc"):
        r = g["r"]
        a0, sweep = (0.0, 360.0) if e["kind"] == "circle" else (g["a0"], (g["a1"] - g["a0"]) % 360)
        da, ds = math.degrees(dash / r), math.degrees(space / r)
        t = 0.0
        while t < sweep - 1e-6:
            t2 = min(t + da, sweep)
            out.append(mk("arc", {"c": g["c"], "r": r, "a0": (a0 + t) % 360, "a1": (a0 + t2) % 360}))
            t = t2 + ds
        return out or [e]
    return [e]


def apply_noise(drawing: dict, rng: np.random.Generator, level: str = "heavy", **overrides):
    cfg = {**LEVELS[level], **overrides}
    d = copy.deepcopy(drawing)
    log = {"level": level, "gaps": 0, "fragments": 0, "extended": 0, "duplicates": 0, "dims_exploded": 0,
           "dashes_exploded": 0, "collapsed": False, "scale_text_dropped": False, "units_header": True}
    ents = d["ents"]

    if cfg.get("explode_dims"):
        new = []
        for e in ents:
            if e["kind"].startswith("dim_"):
                new += explode_dim(e)
                log["dims_exploded"] += 1
            else:
                new.append(e)
        ents = new
    if cfg.get("explode_dashes"):
        new = []
        for e in ents:
            if e["role"] == "hidden" and e["kind"] in ("line", "circle", "arc"):
                pieces = explode_dashes(e)
                log["dashes_exploded"] += 1
                new += pieces
            else:
                new.append(e)
        ents = new

    geo = ("visible", "hidden", "center")
    if cfg.get("frag", 0):
        new = []
        for e in ents:
            if e["kind"] == "line" and e["role"] in geo and rng.random() < cfg["frag"]:
                x0, y0, x1, y1 = e["g"]["p"]
                L = math.hypot(x1 - x0, y1 - y0)
                if L > 8:
                    k = int(rng.integers(2, 5))
                    cuts = np.sort(rng.uniform(0.15, 0.85, k - 1))
                    ts = [0.0, *cuts, 1.0]
                    for a, b in zip(ts[:-1], ts[1:]):
                        n = copy.deepcopy(e)
                        n["g"]["p"] = [x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b]
                        new.append(n)
                    log["fragments"] += 1
                    continue
            new.append(e)
        ents = new
    if cfg.get("overlap", 0) or cfg.get("dup", 0):
        extra = []
        for e in ents:
            if e["kind"] == "line" and e["role"] in geo:
                x0, y0, x1, y1 = e["g"]["p"]
                L = math.hypot(x1 - x0, y1 - y0)
                if L < 3:
                    continue
                ux, uy = (x1 - x0) / L, (y1 - y0) / L
                if rng.random() < cfg.get("overlap", 0):
                    a, b = rng.uniform(0, 0.3, 2)
                    e["g"]["p"] = [x0 - ux * a, y0 - uy * a, x1 + ux * b, y1 + uy * b]
                    log["extended"] += 1
                if rng.random() < cfg.get("dup", 0):
                    n = copy.deepcopy(e)
                    n["g"]["p"] = [v + float(rng.uniform(-0.02, 0.02)) for v in n["g"]["p"]]
                    extra.append(n)
                    log["duplicates"] += 1
        ents += extra
    if cfg.get("gap", 0):
        g = cfg["gap"]
        for e in ents:
            if e["role"] not in geo:
                continue
            if e["kind"] == "line":
                p = e["g"]["p"]
                if math.hypot(p[2] - p[0], p[3] - p[1]) < 4 * g:
                    continue
                for i in (0, 2):
                    if rng.random() < cfg["p_gap"]:
                        r, th = rng.uniform(0.2, 1.0) * g, rng.uniform(0, 2 * math.pi)
                        p[i] += r * math.cos(th)
                        p[i + 1] += r * math.sin(th)
                        log["gaps"] += 1
            elif e["kind"] == "arc":
                for k in ("a0", "a1"):
                    if rng.random() < cfg["p_gap"]:
                        e["g"][k] = (e["g"][k] + math.degrees(rng.uniform(-1, 1) * g / e["g"]["r"])) % 360
                        log["gaps"] += 1
    if cfg.get("collapse"):
        for e in ents:
            e["layer"] = "0"
        log["collapsed"] = True
    if rng.random() < cfg.get("p_scale_drop", 0):
        before = len(ents)
        ents = [e for e in ents if e.get("tag") != "scale"]
        log["scale_text_dropped"] = len(ents) < before
    log["units_header"] = not (rng.random() < cfg.get("p_units_drop", 0))
    d["ents"] = ents
    d["meta"]["units_header"] = log["units_header"]
    return d, log
