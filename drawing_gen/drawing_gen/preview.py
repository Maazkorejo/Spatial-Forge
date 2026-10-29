"""Render a DXF to PNG (own tiny renderer: lines, circles, arcs, text, dashed linetypes, native dimensions)."""
from __future__ import annotations

import math

import cv2
import ezdxf
import numpy as np

COL = {"VISIBLE": (30, 30, 30), "HIDDEN": (160, 90, 20), "CENTER": (40, 40, 200), "DIM": (20, 140, 20), "TITLE": (120, 120, 120)}


def _dashed(img, p0, p1, col, pattern):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-6:
        return
    u, t, i = (p1 - p0) / L, 0.0, 0
    while t < L:
        seg = pattern[i % len(pattern)]
        if i % 2 == 0:
            a, b = p0 + u * t, p0 + u * min(t + seg, L)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, 1, cv2.LINE_AA)
        t += seg
        i += 1


def render_dxf(path: str, out_png: str, ppm: float = 4.0, margin: int = 30) -> str:
    doc = ezdxf.readfile(path)
    ents = []
    for e in doc.modelspace():
        if e.dxftype() == "DIMENSION":
            ents += list(e.virtual_entities())
        else:
            ents.append(e)
    pts = []
    for e in ents:
        t = e.dxftype()
        if t == "LINE":
            pts += [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
        elif t in ("CIRCLE", "ARC"):
            c, r = e.dxf.center, e.dxf.radius
            pts += [(c.x - r, c.y - r), (c.x + r, c.y + r)]
        elif t in ("TEXT", "MTEXT"):
            pts.append((e.dxf.insert.x, e.dxf.insert.y))
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    W, H = int((hi[0] - lo[0]) * ppm) + 2 * margin, int((hi[1] - lo[1]) * ppm) + 2 * margin
    img = np.full((H, W, 3), 255, np.uint8)
    P = lambda x, y: (int((x - lo[0]) * ppm + margin), int(H - ((y - lo[1]) * ppm + margin)))
    for e in ents:
        t = e.dxftype()
        lay = e.dxf.get("layer", "0")
        col = COL.get(lay, (30, 30, 30))
        lt = str(e.dxf.get("linetype", "BYLAYER")).upper()
        if lt == "BYLAYER" and lay in doc.layers:
            lt = doc.layers.get(lay).dxf.linetype.upper()
        pat = {"DASHED": (12, 6), "CENTER": (24, 5, 4, 5)}.get(lt)
        if t == "LINE":
            a, b = P(e.dxf.start.x, e.dxf.start.y), P(e.dxf.end.x, e.dxf.end.y)
            _dashed(img, a, b, col, pat) if pat else cv2.line(img, a, b, col, 1, cv2.LINE_AA)
        elif t in ("CIRCLE", "ARC"):
            c, r = e.dxf.center, e.dxf.radius
            a0, a1 = (0, 360) if t == "CIRCLE" else (e.dxf.start_angle, e.dxf.end_angle)
            if a1 <= a0:
                a1 += 360
            n = max(12, int((a1 - a0) / 4))
            ang = np.radians(np.linspace(a0, a1, n))
            poly = np.array([P(c.x + r * math.cos(a), c.y + r * math.sin(a)) for a in ang], np.int32)
            if pat:
                for i in range(len(poly) - 1):
                    if (i // 2) % 2 == 0:
                        cv2.line(img, tuple(poly[i]), tuple(poly[i + 1]), col, 1, cv2.LINE_AA)
            else:
                cv2.polylines(img, [poly], False, col, 1, cv2.LINE_AA)
        elif t in ("TEXT", "MTEXT"):
            txt = e.dxf.text if t == "TEXT" else e.text
            cv2.putText(img, txt.replace("Ø", "D"), P(e.dxf.insert.x, e.dxf.insert.y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        elif t == "SOLID":
            v = [P(*(getattr(e.dxf, k).x, getattr(e.dxf, k).y)) for k in ("vtx0", "vtx1", "vtx2")]
            cv2.fillPoly(img, [np.array(v, np.int32)], col)
    cv2.imwrite(out_png, img)
    return out_png
