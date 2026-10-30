"""
Safety gate. isValid() is necessary but NOT sufficient: a hole in the wrong place is a perfectly valid solid.
So each edit is also checked against independent expectations:

  errors   (edit rejected, nothing committed)
    * not valid / more than one solid / (nearly) zero volume
    * a feature that had an effect before now has none (the edit pushed it off the part), or a new feature has none
    * a cut removed more volume than its tool could (impossible => something is wrong)
    * geometry outside base block + bosses
    * STEP export -> re-import changes volume by > 1e-4 or is invalid
  warnings (edit allowed, shown to the user)
    * tool only partly in material (overlaps an existing void or leaves the part)
    * fillet/chamfer volume far from the 90-degree-convex-edge estimate (concave / acute edges)
    * overall size differs from the base block (a cut removed an outer extent; use `resize` if intended)
Thresholds are configurable defaults, not standards.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cadquery as cq

from build import FACE, Step
from model import Boss, FeatureTree

NO_EFFECT = 0.02          # actual/ideal below this = the feature does nothing
PARTIAL = 0.90            # below this = tool only partly in material
OVER = 1.001              # a cut cannot remove more than its tool volume


@dataclass
class GateReport:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def expected_bbox(tree: FeatureTree) -> tuple:
    lo, hi = [0.0, 0.0, 0.0], [tree.base.W, tree.base.D, tree.base.H]
    for f in tree.features:
        if isinstance(f, Boss):
            _, _, na, sg = FACE[f.face]
            if sg > 0:
                hi[na] = max(hi[na], (tree.base.W, tree.base.D, tree.base.H)[na] + f.height)
            else:
                lo[na] = min(lo[na], -f.height)
    return tuple(lo), tuple(hi)


def bbox_of(solid: cq.Workplane) -> tuple:
    b = solid.val().BoundingBox()
    return (b.xmin, b.ymin, b.zmin), (b.xmax, b.ymax, b.zmax)


def step_roundtrip(solid: cq.Workplane) -> tuple:
    """(rel volume error, valid) after STEP export -> import."""
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    try:
        cq.exporters.export(solid, path)
        back = cq.importers.importStep(path)
        v0, v1 = solid.val().Volume(), back.val().Volume()
        return abs(v1 - v0) / max(v0, 1e-12), bool(back.val().isValid())
    finally:
        os.remove(path)


def check(tree: FeatureTree, solid: cq.Workplane, trace: list, prev_trace: Optional[list] = None,
          roundtrip: bool = True) -> GateReport:
    errors, warnings = [], []
    prev = {s.fid: s for s in (prev_trace or [])}
    base_vol = tree.base.W * tree.base.D * tree.base.H
    try:
        shape = solid.val()
        n_solids = len(solid.solids().vals())
        vol = float(shape.Volume())
        valid = bool(shape.isValid())
    except Exception:
        return GateReport(False, ["The edit removed the entire part (empty result)."], [], {"volume": 0.0, "n_solids": 0})
    if n_solids == 0:
        return GateReport(False, ["The edit removed the entire part (empty result)."], [], {"volume": 0.0, "n_solids": 0})

    if not valid:
        errors.append("The result is not a valid B-rep solid.")
    if n_solids != 1:
        errors.append(f"The edit splits the part into {n_solids} separate solids.")
    if vol < 1e-4 * base_vol:
        errors.append("The part (almost) vanished: remaining volume is under 0.01% of the base block.")

    for s in trace:
        if s.ideal is None or s.ideal <= 0:
            if s.role == "shell" and s.actual <= 0:
                errors.append(f"{s.fid}: shell removed no material.")
            continue
        ratio = s.actual / s.ideal
        p = prev.get(s.fid)
        p_ratio = (p.actual / p.ideal) if (p and p.ideal) else None
        if s.role in ("cut", "add", "edge"):
            if ratio < NO_EFFECT:
                if p_ratio is None:
                    errors.append(f"{s.fid}: has no effect (it misses the part or sits entirely in an existing void).")
                elif p_ratio >= NO_EFFECT:
                    errors.append(f"{s.fid}: would no longer touch the part after this edit (it had an effect before).")
                else:
                    warnings.append(f"{s.fid}: still has no effect.")
                continue
        if s.role == "cut" and ratio > OVER:
            errors.append(f"{s.fid}: removed {s.actual:.1f} mm³ but its tool holds only {s.ideal:.1f} mm³ (volume mismatch).")
        elif s.role == "cut" and ratio < PARTIAL:
            warnings.append(f"{s.fid}: tool is only partly in material (removed {s.actual:.1f} of {s.ideal:.1f} mm³): it "
                            f"overlaps an existing void or leaves the part.")
        elif s.role == "add" and ratio < PARTIAL:
            warnings.append(f"{s.fid}: added {s.actual:.1f} of {s.ideal:.1f} mm³ (partly overlaps existing material).")
        elif s.role == "edge" and not (0.5 <= ratio <= 1.2):
            warnings.append(f"{s.fid}: volume change {s.actual:.1f} mm³ vs {s.ideal:.1f} expected for convex 90° edges "
                            f"(concave or acute edges are involved).")

    elo, ehi = expected_bbox(tree)
    alo, ahi = bbox_of(solid)
    tol = 1e-4 + 1e-6 * max(ehi)
    if any(a < e - tol for a, e in zip(alo, elo)) or any(a > e + tol for a, e in zip(ahi, ehi)):
        errors.append("Geometry lies outside the base block plus bosses.")
    elif any(abs(a - e) > tol for a, e in zip(ahi, ehi)) or any(abs(a - e) > tol for a, e in zip(alo, elo)):
        warnings.append("Overall size differs from the base block: a cut removed an outer extent (use 'resize' if intended).")

    rt_err = None
    if roundtrip and valid and n_solids == 1:
        try:
            rt_err, rt_valid = step_roundtrip(solid)
            if not rt_valid or rt_err > 1e-4:
                errors.append(f"STEP export/import changes the part (volume error {rt_err:.1e}, valid={rt_valid}).")
        except Exception as e:
            errors.append(f"STEP export/import failed: {type(e).__name__}")

    stats = {"volume": vol, "valid": valid, "n_solids": n_solids, "bbox_lo": alo, "bbox_hi": ahi,
             "size": tuple(h - l for l, h in zip(alo, ahi)), "step_roundtrip_rel_err": rt_err}
    return GateReport(not errors, errors, warnings, stats)
