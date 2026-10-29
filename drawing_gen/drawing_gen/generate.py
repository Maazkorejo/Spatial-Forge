"""
Benchmark case generator.   python generate.py OUTDIR N [--levels clean,light,heavy] [--seed0 0] [--hull]

Per case  OUTDIR/case_XXXX_<level>.{dxf, gt.json, step}:
  dxf      the (possibly dirty) drawing
  step     ground-truth solid
  gt.json  spec + features, scale / standard / view layout, true dimensions, noise log, CLEAN entity list with roles
"""
from __future__ import annotations

import argparse
import json
import os

import cadquery as cq
import numpy as np

from draw import SCALES, build_drawing
from dxf_io import write_dxf
from noise import apply_noise
from parts import make_part


def make_case(seed: int, level: str = "clean", family: str | None = None, standard: str | None = None,
              scale: str | None = None, with_hull: bool = False):
    rng = np.random.default_rng(seed)
    body, spec = make_part(rng, family)
    standard = standard or ("third" if rng.random() < 0.7 else "first")
    scale = scale or str(rng.choice(["1:1", "1:1", "1:1", "1:2", "2:1"]))
    clean = build_drawing(body, spec, standard, scale, name=f"PART-{seed:04d}")
    noisy, log = apply_noise(clean, np.random.default_rng(seed + 10_000), level)
    if with_hull:
        from metrics import hull_iou
        spec["hull_iou"] = hull_iou(body.val())["iou"]
    return {"seed": seed, "level": level, "solid": body, "spec": spec, "clean": clean, "noisy": noisy, "noise_log": log}


def save_case(case: dict, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    name = f"case_{case['seed']:04d}_{case['level']}"
    dxf = write_dxf(case["noisy"], os.path.join(outdir, name + ".dxf"), units_header=case["noisy"]["meta"].get("units_header", True))
    step = os.path.join(outdir, name + ".step")
    cq.exporters.export(case["solid"], step)
    gt = {"seed": case["seed"], "level": case["level"], "spec": case["spec"], "meta": case["clean"]["meta"],
          "noise_log": case["noise_log"], "clean_entities": case["clean"]["ents"]}
    with open(os.path.join(outdir, name + ".gt.json"), "w") as fh:
        json.dump(gt, fh)
    return {"dxf": dxf, "step": step, "gt": os.path.join(outdir, name + ".gt.json")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("n", type=int)
    ap.add_argument("--levels", default="clean,light,heavy")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--hull", action="store_true", help="also compute visual-hull IoU into the spec (slower)")
    a = ap.parse_args()
    for s in range(a.seed0, a.seed0 + a.n):
        for lv in a.levels.split(","):
            paths = save_case(make_case(s, lv, with_hull=a.hull), a.outdir)
        print("seed", s, "->", os.path.basename(paths["dxf"]))
