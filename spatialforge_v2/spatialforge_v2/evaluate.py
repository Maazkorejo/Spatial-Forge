"""
Evaluation harness. You cannot claim fewer hallucinations without measuring.

Metrics per case (pred vs ground-truth Assembly):
  part_f1        F1 over (kind, parent kind, face) relation tuples: right parts attached the right way
  silhouette_iou IoU of the two silhouettes under the SAME camera (ground-truth view)
  dim_err        mean relative error of dims, normalised by overall size (parts matched by id; NaN if ids differ)
  valid          geometry_report has no errors
  feature_p/r    features matched on (kind, part kind, face) with |du|+|dv| <= tol
Directory mode:  python evaluate.py DIR     expects NAME.gt.json + NAME.pred.json (+ optional NAME.cam.json)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import numpy as np

import fit as F
from schema import Assembly
from solver import geometry_report, resolve


def _rel_tuples(a: Assembly) -> Counter:
    by = {p.id: p for p in a.parts}
    return Counter((p.kind, by[p.placement.parent].kind if p.placement else "root",
                    p.placement.face if p.placement else "-", p.op) for p in a.parts)


def _f1(pred: Counter, gt: Counter) -> float:
    tp = sum((pred & gt).values())
    if not pred or not gt:
        return 0.0
    p, r = tp / sum(pred.values()), tp / sum(gt.values())
    return 0.0 if tp == 0 else 2 * p * r / (p + r)


def silhouette_iou(pred: Assembly, gt: Assembly, cam: F.Camera, S: int = 200) -> float:
    polys = F._project(resolve(gt), cam)
    allp = np.vstack([p for p, _ in polys])
    w, h = allp.max(0) - allp.min(0)
    big = np.ones((int(1000 * h / max(w, h)) + 2, int(1000 * w / max(w, h)) + 2), np.uint8)
    t0 = F.Target(big, S=S)
    gt_mask = F.render(gt, cam, t0)
    tgt = F.Target(gt_mask, S=S)
    best = 0.0
    for c in (cam,):
        img = F.render(pred, c, tgt)
        if img is not None:
            best = max(best, F.score(img, tgt)[0])
    return best


def dim_error(pred: Assembly, gt: Assembly) -> float:
    pm, gm = {p.id: p for p in pred.parts}, {p.id: p for p in gt.parts}
    if set(pm) != set(gm):
        return float("nan")
    errs = []
    norm = lambda a: a.parts[0].dims()[next(iter(a.parts[0].dims()))].value
    kp, kg = norm(pred), norm(gt)
    for pid, g in gm.items():
        for f, d in g.dims().items():
            errs.append(abs((pm[pid].dims()[f].value / kp) / (d.value / kg) - 1))
    return float(np.mean(errs))


def feature_pr(pred: Assembly, gt: Assembly, tol: float = 0.2) -> tuple:
    kind_of = lambda a: {p.id: p.kind for p in a.parts}
    kp, kg = kind_of(pred), kind_of(gt)
    used, tp = set(), 0
    for f in pred.features:
        for i, g in enumerate(gt.features):
            if i in used:
                continue
            if (f.kind, kp[f.on.part], f.on.face) == (g.kind, kg[g.on.part], g.on.face) \
                    and abs(f.on.u - g.on.u) + abs(f.on.v - g.on.v) <= tol:
                used.add(i)
                tp += 1
                break
    p = tp / len(pred.features) if pred.features else 0.0
    r = tp / len(gt.features) if gt.features else 0.0
    return p, r


def evaluate_case(pred: Assembly, gt: Assembly, cam: F.Camera) -> dict:
    p, r = feature_pr(pred, gt)
    return {"part_f1": _f1(_rel_tuples(pred), _rel_tuples(gt)),
            "silhouette_iou": silhouette_iou(pred, gt, cam),
            "dim_err": dim_error(pred, gt),
            "valid": not geometry_report(pred)["errors"],
            "feature_p": p, "feature_r": r}


def evaluate_dir(path: str) -> dict:
    rows = {}
    for gt_path in sorted(glob.glob(os.path.join(path, "*.gt.json"))):
        name = os.path.basename(gt_path)[:-8]
        pred_path = os.path.join(path, f"{name}.pred.json")
        if not os.path.exists(pred_path):
            continue
        gt = Assembly.model_validate_json(open(gt_path).read())
        try:
            pred = Assembly.model_validate_json(open(pred_path).read())
        except Exception:
            rows[name] = {"valid": False}
            continue
        cp = os.path.join(path, f"{name}.cam.json")
        cam = F.Camera(**json.load(open(cp))) if os.path.exists(cp) else F.Camera(
            gt.view_hint.azimuth_deg or -35, gt.view_hint.elevation_deg or 25)
        rows[name] = evaluate_case(pred, gt, cam)
    if rows:
        keys = ["part_f1", "silhouette_iou", "dim_err", "feature_p", "feature_r"]
        rows["MEAN"] = {k: float(np.nanmean([r[k] for r in rows.values() if k in r])) for k in keys}
        rows["MEAN"]["valid_rate"] = float(np.mean([r.get("valid", False) for n, r in rows.items() if n != "MEAN"]))
    return rows


if __name__ == "__main__":
    out = evaluate_dir(sys.argv[1] if len(sys.argv) > 1 else ".")
    for k, v in out.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
