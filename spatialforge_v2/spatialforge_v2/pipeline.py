"""
Pipeline v2:
  mask -> N VLM candidates -> validate -> fit -> (per-part IoU if part masks) -> pick best
       -> structural repair loop -> feature detection + back-projection -> multi-view check -> Result

    result = run(image_bgr, vlm, mask=sam_mask, detector=my_detector, part_masks={...})

`mask` should come from SAM / SAM 2. `detector(image_bgr) -> list[features.Detection]` wraps a grounding
model (Florence-2 / GroundingDINO / OWL-ViT). `part_masks` is optional {part_id: mask in original-image px};
it only makes sense when part ids can be tied to SAM masks (e.g. the user clicked the parts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import fit as fitmod
from features import place_features
from fit import Target, apply_fit, diagnose, make_overlay
from geometry import part_mesh
from multiview import multiview_check
from prompt import build_error_feedback, build_repair_prompt, build_system_prompt, build_user_prompt
from raster import part_visibility_iou, rasterize
from schema import Assembly, json_schema, parse_assembly
from solver import geometry_report, resolve


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes()


def extract_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Fallback only. Otsu on a line drawing + close + fill the largest external contour."""
    g = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return out


@dataclass
class Candidate:
    asm: Assembly
    fit: fitmod.FitResult
    warnings: list
    part_iou: dict
    score: float


@dataclass
class Result:
    status: str                          # "ok" | "needs_review" | "failed"
    assembly: Optional[Assembly] = None
    iou: float = 0.0
    part_iou: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    rejected_features: list = field(default_factory=list)
    view_check: Optional[dict] = None
    log: list = field(default_factory=list)
    overlay_png: Optional[bytes] = None


def propose(vlm, images: list, user: str, system: str, temperature: float,
            max_retries: int = 2, log: Callable = print) -> Optional[Assembly]:
    """One VLM call with parse/validate retry. Errors are fed back verbatim."""
    prompt = user
    for attempt in range(max_retries + 1):
        raw = vlm.complete(system, prompt, images, json_schema(), temperature)
        asm, errs = parse_assembly(raw)
        if asm is not None:
            rep = geometry_report(asm)
            if not rep["errors"]:
                return asm
            errs = rep["errors"]
        log(f"  attempt {attempt}: rejected ({len(errs)} errors)")
        prompt = user + "\n\n" + build_error_feedback(errs)
    return None


def _frame(asm: Assembly, cam, target: Target, ov=None):
    return fitmod.make_frame(fitmod._project(resolve(asm, ov), cam), target)


def _part_ious(asm: Assembly, fr: fitmod.FitResult, target: Target, part_masks: Optional[dict]) -> dict:
    if not part_masks:
        return {}
    solids = resolve(asm)
    meshes = {p: part_mesh(s) for p, s in solids.items() if s.op == "union"}
    frame = _frame(asm, fr.camera, target)
    ids, names, _, _ = rasterize(meshes, fr.camera, frame)
    canvas = {p: target.warp(m) for p, m in part_masks.items()}
    return part_visibility_iou(ids, names, canvas)


def evaluate(asm: Assembly, target: Target, part_masks=None, **fit_kw) -> Optional[Candidate]:
    rep = geometry_report(asm)
    if rep["errors"]:
        return None
    fr = fitmod.fit(asm, target, **fit_kw)
    fitted = apply_fit(asm, fr)
    pious = _part_ious(fitted, fr, target, part_masks)
    base = fr.iou if not pious else 0.5 * fr.iou + 0.5 * float(np.mean(list(pious.values())))
    return Candidate(fitted, fr, rep["warnings"], pious, base - 0.02 * len(rep["warnings"]))


def run(image_bgr: np.ndarray, vlm, mask: Optional[np.ndarray] = None,
        detector: Optional[Callable] = None, part_masks: Optional[dict] = None,
        n_candidates: int = 3, min_iou: float = 0.85, max_repairs: int = 2,
        check_views: bool = True, anchor_hint: Optional[str] = None,
        log: Callable = print, **fit_kw) -> Result:
    target = Target(mask if mask is not None else extract_mask(image_bgr))
    png = encode_png(image_bgr)
    system, user = build_system_prompt(), build_user_prompt(anchor_hint)

    cands: list = []
    for i in range(n_candidates):
        log(f"candidate {i + 1}/{n_candidates}")
        asm = propose(vlm, [png], user, system, 0.2 if i == 0 else 0.7, log=log)
        c = evaluate(asm, target, part_masks, **fit_kw) if asm else None
        if c:
            log(f"  IoU {c.fit.init_iou:.2f} -> {c.fit.iou:.2f}, {len(c.asm.parts)} parts")
            cands.append(c)
    if not cands:
        return Result("failed", log=["no valid candidate from the VLM"])
    best = max(cands, key=lambda c: c.score)

    view = None
    for r in range(max_repairs + 1):
        if check_views:
            view = multiview_check(vlm, best.asm, png)
        views_bad = bool(view) and view["plausible"] is False
        if best.fit.iou >= min_iou and not views_bad:
            break
        if r == max_repairs:
            break
        log(f"repair {r + 1}: IoU {best.fit.iou:.2f}, view check {view and view['score']}")
        diag = diagnose(best.fit.rendered, target)
        weak = [p for p, s in best.fit.sens_part.items() if s < fitmod.SENS_THRESHOLD]
        prompt = build_repair_prompt(best.asm, best.fit.iou, diag, weak, view["issues"] if views_bad else None)
        asm = propose(vlm, [png, make_overlay(best.fit.rendered, target)], prompt, system, 0.3, log=log)
        c = evaluate(asm, target, part_masks, **fit_kw) if asm else None
        if c and c.score > best.score:
            best = c

    # features: detected, then back-projected onto the fitted body (never VLM-invented)
    final, rejected = best.asm, []
    if detector is not None:
        frame = _frame(final, best.fit.camera, target)
        feats, rejected = place_features(final, best.fit.camera, frame, target, detector(image_bgr))
        final = final.model_copy(update={"features": feats})
        log(f"features: {len(feats)} placed, {len(rejected)} rejected")

    ok = best.fit.iou >= min_iou and not (view and view["plausible"] is False)
    return Result("ok" if ok else "needs_review", final, best.fit.iou, best.part_iou, best.warnings,
                  rejected, view, [], make_overlay(best.fit.rendered, target))
