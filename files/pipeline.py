"""
Pipeline: sketch -> mask -> N VLM candidates -> validate -> fit -> pick best -> (repair) -> result.

    result = run(image_bgr, vlm, mask=sam_mask)

`mask` should come from SAM / SAM 2 (whole-object mask). extract_mask() is a crude fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import fit as fitmod
from fit import Target, apply_fit, diagnose, make_overlay, render
from prompt import build_error_feedback, build_repair_prompt, build_system_prompt, build_user_prompt
from schema import Assembly, json_schema, parse_assembly
from solver import geometry_report


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
    score: float


@dataclass
class Result:
    status: str                       # "ok" | "needs_review" | "failed"
    assembly: Optional[Assembly] = None
    iou: float = 0.0
    warnings: list = field(default_factory=list)
    log: list = field(default_factory=list)
    overlay_png: Optional[bytes] = None


def propose(vlm, images: list[bytes], user: str, system: str, temperature: float,
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


def evaluate(asm: Assembly, target: Target, **fit_kw) -> Optional[Candidate]:
    rep = geometry_report(asm)
    if rep["errors"]:
        return None
    fr = fitmod.fit(asm, target, **fit_kw)
    s = fr.iou - 0.02 * len(rep["warnings"])
    return Candidate(apply_fit(asm, fr), fr, rep["warnings"], s)


def run(image_bgr: np.ndarray, vlm, mask: Optional[np.ndarray] = None, n_candidates: int = 3,
        min_iou: float = 0.85, max_repairs: int = 2, anchor_hint: Optional[str] = None,
        log: Callable = print, **fit_kw) -> Result:
    target = Target(mask if mask is not None else extract_mask(image_bgr))
    png = encode_png(image_bgr)
    system = build_system_prompt()
    user = build_user_prompt(anchor_hint)

    cands: list[Candidate] = []
    for i in range(n_candidates):
        log(f"candidate {i + 1}/{n_candidates}")
        asm = propose(vlm, [png], user, system, 0.2 if i == 0 else 0.7, log=log)
        if asm is None:
            continue
        c = evaluate(asm, target, **fit_kw)
        if c:
            log(f"  IoU {c.fit.init_iou:.2f} -> {c.fit.iou:.2f}, {len(c.asm.parts)} parts")
            cands.append(c)
    if not cands:
        return Result("failed", log=["no valid candidate from the VLM"])

    best = max(cands, key=lambda c: c.score)
    for r in range(max_repairs):
        if best.fit.iou >= min_iou:
            break
        log(f"repair {r + 1}: IoU {best.fit.iou:.2f} < {min_iou}")
        img = best.fit.rendered
        diag = diagnose(img, target)
        weak = [p for p, s in best.fit.sens_part.items() if s < fitmod.SENS_THRESHOLD]
        prompt = build_repair_prompt(best.asm, best.fit.iou, diag, weak)
        asm = propose(vlm, [png, make_overlay(img, target)], prompt, system, 0.3, log=log)
        c = evaluate(asm, target, **fit_kw) if asm else None
        if c and c.score > best.score:
            best = c

    overlay = make_overlay(best.fit.rendered, target) if best.fit.rendered is not None else None
    return Result("ok" if best.fit.iou >= min_iou else "needs_review",
                  best.asm, best.fit.iou, best.warnings, [], overlay)
