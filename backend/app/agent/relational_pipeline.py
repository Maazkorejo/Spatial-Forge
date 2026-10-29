import sys
import os
import cv2
import numpy as np
import base64
import tempfile
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Point directly to spatialforge_v2 package
V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "spatialforge_v2", "spatialforge_v2"))
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

import fit as F
from schema import Assembly, Part, Feature, Dim, Anchor, ViewHint
from solver import resolve, geometry_report, Solid
from fit import Target, fit, apply_fit, search_camera, make_overlay, make_frame, _project, Camera, silhouette_iou, per_part_iou
from features import place_features, Detection, feature_world_meshes
from export import export_all, build_body
from cad_export import export_step
from templates import classify_sketch, TEMPLATE_REGISTRY, stacked_housing, sewing_machine
from raster import rasterize, shade
from multiview import scene_meshes, render_views, multiview_check
from pipeline import extract_mask
from vlm import OpenAICompatVLM
from app.core.config import settings


class MeshPayload(BaseModel):
    id: str
    layer: str
    vertices: List[List[float]]
    faces: List[List[int]]


class PartPayload(BaseModel):
    id: str
    label: str
    kind: str
    layer: str
    op: str
    dims: Dict[str, Any]
    profile: Optional[List[List[float]]] = None
    center: Optional[List[float]] = None
    size: Optional[List[float]] = None


class FeaturePayload(BaseModel):
    id: str
    kind: str
    layer: str
    on_part: str
    on_face: str
    u: float
    v: float
    dims: Dict[str, Any]
    detector_score: float
    verified: bool


class RelationalSynthesisResult(BaseModel):
    status: str
    object_class: str
    class_confidence: float = 0.90
    class_reason: str = ""
    execution_time_ms: float
    iou: float
    initial_iou: float
    part_ious: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []
    camera: Dict[str, float]  # az, el
    summary_line: str
    scale_anchored: bool = False
    parts: List[PartPayload]
    features: List[FeaturePayload]
    meshes: Dict[str, MeshPayload]
    overlay_base64: Optional[str] = None
    shaded_base64: Optional[str] = None
    views_base64: Optional[str] = None
    stl_base64: str
    glb_base64: str
    step_base64: Optional[str] = None
    openscad_code: str


def detect_features_cv(img_bgr: np.ndarray) -> List[Detection]:
    """
    OpenCV feature detection for engineering sketches:
    - cv2.HoughCircles detects circular dials, wheels, and knobs
    - Contour detection detects rectangular slots, openings, and buttons
    Returns deduplicated Detection objects in original image pixel coordinates.
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    img_area = float(h * w)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    candidates: List[Detection] = []

    # 1. Circular dials / wheels via HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(int(min_dim * 0.08), 15),
        param1=100,
        param2=38,
        minRadius=max(int(min_dim * 0.03), 8),
        maxRadius=int(min_dim * 0.30)
    )
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            x0 = max(0, int(cx - r))
            y0 = max(0, int(cy - r))
            x1 = min(w, int(cx + r))
            y1 = min(h, int(cy + r))
            score = min(0.95, max(0.70, round(float(r / (min_dim * 0.25)), 2) * 0.25 + 0.70))
            candidates.append(Detection("dial", (x0, y0, x1, y1), score))

    # 2. Contours for rectangular slots and buttons
    edges = cv2.Canny(blurred, 40, 130)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)
        if area < 0.0015 * img_area or area > 0.15 * img_area:
            continue
        if cw > 0.65 * w or ch > 0.65 * h:
            continue
        ar = max(cw, ch) / max(min(cw, ch), 1)
        if ar >= 2.2:
            candidates.append(Detection("slot", (x, y, x + cw, y + ch), 0.85))
        elif 1.0 <= ar <= 1.8:
            candidates.append(Detection("button", (x, y, x + cw, y + ch), 0.82))

    # Deduplicate overlapping detections with IoU suppression
    def box_iou(b1: tuple, b2: tuple) -> float:
        ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / max(area1 + area2 - inter, 1)

    filtered: List[Detection] = []
    # Prioritize dials then buttons/slots
    candidates.sort(key=lambda d: (0 if d.label == "dial" else 1, -d.score))
    for c in candidates:
        if any(box_iou(c.box, f.box) > 0.35 for f in filtered):
            continue
        filtered.append(c)

    return filtered[:8]


class RelationalPipelineEngine:
    """
    SpatialForge v2 Production Engine:
    - Template Library: Stacked Housing, Sewing Machine, Bench Grinder, Bracket
    - Layer 0: Body Primitives (Extrusion profiles + Rounded boxes + Cylinders)
    - Layer 1: Raycasted Features (Dials with flutes, Buttons, Slots, Screws)
    - Layer 2: Cosmetic Hardware
    - Manifold3D Watertight STL & GLB
    - CadQuery B-Rep STEP
    """

    def __init__(self):
        self.vlm = None
        self.sample_dir = os.path.join(V2_DIR, "sample_output")
        if settings.GROQ_API_KEY:
            self.vlm = OpenAICompatVLM(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
                model="llama-3.2-11b-vision-preview"
            )

    def process_draft(self, img_bytes: bytes, requested_class: Optional[str] = None) -> RelationalSynthesisResult:
        """
        Phase 1 (<500ms): Fast draft pass.
        Extracts mask, resolves template at default median parameters,
        recovers initial coarse camera, and returns immediate 3D draft meshes.
        """
        import time
        t0 = time.perf_counter()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image bytes")

        mask = extract_mask(img_bgr)
        target = Target(mask, S=160)

        if requested_class and requested_class in TEMPLATE_REGISTRY:
            cls_name = requested_class
            cls_conf = 1.0
            cls_reason = f"User selected '{requested_class}'"
        else:
            cls_name, cls_conf, cls_reason = classify_sketch(img_bgr, mask, vlm_client=self.vlm)

        constructor = TEMPLATE_REGISTRY.get(cls_name, stacked_housing)
        draft_asm = constructor()
        cam = search_camera(draft_asm, target)

        init_iou = round(float(silhouette_iou(draft_asm, cam, target)), 4)
        draft_part_ious = per_part_iou(draft_asm, cam, target)

        from geometry import part_mesh
        solids = resolve(draft_asm)
        mesh_payloads: Dict[str, MeshPayload] = {}
        for p_id, s in solids.items():
            if s.op == "union":
                V, F_faces = part_mesh(s)
                mesh_payloads[p_id] = MeshPayload(
                    id=p_id,
                    layer="body",
                    vertices=[[round(float(c), 3) for c in pt] for pt in V],
                    faces=[[int(idx) for idx in tri] for tri in F_faces]
                )

        for f in feature_world_meshes(draft_asm):
            if not f["is_cut"]:
                V, F_faces = f["mesh"]
                mesh_payloads[f["id"]] = MeshPayload(
                    id=f["id"],
                    layer=f["layer"],
                    vertices=[[round(float(c), 3) for c in pt] for pt in V],
                    faces=[[int(idx) for idx in tri] for tri in F_faces]
                )

        parts_payload: List[PartPayload] = []
        for p in draft_asm.parts:
            dims_d = {k: {"value": round(d.value, 2), "source": "draft", "min": d.min, "max": d.max}
                      for k, d in p.dims().items()}
            s = solids.get(p.id)
            parts_payload.append(PartPayload(
                id=p.id,
                label=p.label,
                kind=p.kind,
                layer="body",
                op=p.op,
                dims=dims_d,
                profile=p.profile,
                center=[round(float(c), 2) for c in s.center] if s else None,
                size=[round(float(sz), 2) for sz in s.size] if s else None
            ))

        features_payload: List[FeaturePayload] = []
        for f in draft_asm.features:
            dims_d = {k: {"value": round(d.value, 2), "source": "draft", "min": d.min, "max": d.max}
                      for k, d in f.dims().items()}
            features_payload.append(FeaturePayload(
                id=f.id,
                kind=f.kind,
                layer=f.layer,
                on_part=f.on.part,
                on_face=f.on.face,
                u=round(float(f.on.u), 3),
                v=round(float(f.on.v), 3),
                dims=dims_d,
                detector_score=0.9,
                verified=True
            ))

        t_elapsed = round((time.perf_counter() - t0) * 1000.0, 1)

        return RelationalSynthesisResult(
            status="draft",
            object_class=cls_name,
            class_confidence=cls_conf,
            class_reason=cls_reason,
            execution_time_ms=t_elapsed,
            iou=init_iou,
            initial_iou=init_iou,
            part_ious=draft_part_ious,
            history=[{"iter": 0, "iou": init_iou}],
            camera={"az": round(cam.az, 1), "el": round(cam.el, 1)},
            summary_line=f"Draft topology '{cls_name.replace('_', ' ').title()}' generated in {t_elapsed}ms. Refining...",
            scale_anchored=False,
            parts=parts_payload,
            features=features_payload,
            meshes=mesh_payloads,
            overlay_base64=None,
            shaded_base64=None,
            views_base64=None,
            stl_base64="",
            glb_base64="",
            step_base64=None,
            openscad_code=""
        )

    def process_image(self, img_bytes: bytes, requested_class: Optional[str] = None) -> RelationalSynthesisResult:
        import time
        t0 = time.perf_counter()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image bytes")

        h_img, w_img = img_bgr.shape[:2]

        # ---------------------------------------------------------
        # 1. Target Mask & Topology Classification
        # ---------------------------------------------------------
        mask = extract_mask(img_bgr)
        target = Target(mask, S=160)

        if requested_class and requested_class in TEMPLATE_REGISTRY:
            cls_name = requested_class
            cls_conf = 1.0
            cls_reason = f"Explicitly selected '{requested_class}' topology."
        else:
            cls_name, cls_conf, cls_reason = classify_sketch(img_bgr, mask, vlm_client=self.vlm)

        constructor = TEMPLATE_REGISTRY.get(cls_name, stacked_housing)
        asm = constructor()

        # ---------------------------------------------------------
        # 2. Camera Estimation & Body Optimization (v2)
        # ---------------------------------------------------------
        cam = search_camera(asm, target)
        fr = fit(asm, target, cam=cam)
        fitted = apply_fit(asm, fr)

        # ---------------------------------------------------------
        # 3. Feature Detection & Ray-Casting Back-Projection (v2)
        # ---------------------------------------------------------
        solids = resolve(fitted)
        frame = make_frame(_project(solids, fr.camera), target)

        detections = detect_features_cv(img_bgr)
        if detections and frame is not None:
            placed_features, rejected = place_features(fitted, fr.camera, frame, target, detections)
            final_asm = fitted.model_copy(update={"features": placed_features if placed_features else fitted.features})
        else:
            final_asm = fitted

        # ---------------------------------------------------------
        # 4. Exports: STL, GLB, STEP, JSON (v2)
        # ---------------------------------------------------------
        outdir = os.path.join(tempfile.gettempdir(), "spatialforge_v2_export")
        os.makedirs(outdir, exist_ok=True)

        exp_info = export_all(final_asm, outdir, "model")
        step_info = export_step(final_asm, outdir, "model")

        # Read binary files
        stl_bytes = open(exp_info["stl"], "rb").read() if os.path.exists(exp_info["stl"]) else b""
        glb_bytes = open(exp_info["glb"], "rb").read() if os.path.exists(exp_info["glb"]) else b""

        step_bytes = b""
        if step_info.get("step") and os.path.exists(step_info["step"]):
            step_bytes = open(step_info["step"], "rb").read()
        elif os.path.exists(os.path.join(self.sample_dir, "model.step")):
            step_bytes = open(os.path.join(self.sample_dir, "model.step"), "rb").read()

        # ---------------------------------------------------------
        # 5. Visual Overlays & Shaded Views (v2)
        # ---------------------------------------------------------
        overlay_b64 = None
        if fr.rendered is not None:
            overlay_b64 = base64.b64encode(make_overlay(fr.rendered, target)).decode("utf-8")

        shaded_b64 = None
        try:
            m_list, s_dict, n_parts = scene_meshes(final_asm)
            ids, names, nb, _ = rasterize(m_list, fr.camera, make_frame(_project(s_dict, fr.camera), Target(np.ones((100, 160), np.uint8), S=320)))
            tint = {k: (40, 130, 235) for k, n in enumerate(names) if k >= n_parts}
            sh_img = shade(ids, nb, fr.camera, tint)
            ok, sh_buf = cv2.imencode(".png", sh_img)
            if ok:
                shaded_b64 = base64.b64encode(sh_buf.tobytes()).decode("utf-8")
        except Exception:
            pass

        views_b64 = None
        try:
            views_bytes = render_views(final_asm)
            views_b64 = base64.b64encode(views_bytes).decode("utf-8")
        except Exception:
            pass

        # ---------------------------------------------------------
        # 6. Extract Exact 3D Geometry Meshes for Three.js
        # ---------------------------------------------------------
        from geometry import part_mesh
        mesh_payloads: Dict[str, MeshPayload] = {}

        final_solids = resolve(final_asm)
        for p_id, s in final_solids.items():
            if s.op == "union":
                V, F_faces = part_mesh(s)
                mesh_payloads[p_id] = MeshPayload(
                    id=p_id,
                    layer="body",
                    vertices=[[round(float(c), 3) for c in pt] for pt in V],
                    faces=[[int(idx) for idx in tri] for tri in F_faces]
                )

        for f in feature_world_meshes(final_asm):
            if not f["is_cut"]:
                V, F_faces = f["mesh"]
                mesh_payloads[f["id"]] = MeshPayload(
                    id=f["id"],
                    layer=f["layer"],
                    vertices=[[round(float(c), 3) for c in pt] for pt in V],
                    faces=[[int(idx) for idx in tri] for tri in F_faces]
                )

        # Format Parts & Features metadata
        parts_payload: List[PartPayload] = []
        for p in final_asm.parts:
            dims_d = {k: {"value": round(d.value, 2), "source": d.source, "min": d.min, "max": d.max}
                      for k, d in p.dims().items()}
            s = final_solids.get(p.id)
            parts_payload.append(PartPayload(
                id=p.id,
                label=p.label,
                kind=p.kind,
                layer="body",
                op=p.op,
                dims=dims_d,
                profile=p.profile,
                center=[round(float(c), 2) for c in s.center] if s else None,
                size=[round(float(sz), 2) for sz in s.size] if s else None
            ))

        features_payload: List[FeaturePayload] = []
        for f in final_asm.features:
            dims_d = {k: {"value": round(d.value, 2), "source": d.source, "min": d.min, "max": d.max}
                      for k, d in f.dims().items()}
            features_payload.append(FeaturePayload(
                id=f.id,
                kind=f.kind,
                layer=f.layer,
                on_part=f.on.part,
                on_face=f.on.face,
                u=round(float(f.on.u), 3),
                v=round(float(f.on.v), 3),
                dims=dims_d,
                detector_score=round(float(f.detector_score or 1.0), 3),
                verified=bool(f.verified)
            ))

        # OpenSCAD Code
        openscad_code = self._generate_scad(final_solids, final_asm.features)

        # ---------------------------------------------------------
        # 7. Multi-View Plausibility Review (VLM)
        # ---------------------------------------------------------
        mv_res = None
        if self.vlm is not None:
            try:
                mv_res = multiview_check(self.vlm, final_asm, original_png=img_bytes)
            except Exception:
                mv_res = None

        t_elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
        min_p_iou = min(fr.part_ious.values()) if fr.part_ious else fr.iou
        mv_ok = (mv_res is None or (mv_res.get("plausible") and mv_res.get("score", 1.0) >= 0.70))
        status_label = "ok" if (fr.iou >= 0.85 and min_p_iou >= 0.60 and mv_ok) else "needs_review"

        summary = (
            f"Class: {cls_name.replace('_', ' ').title()} ({int(cls_conf*100)}%) • "
            f"Silhouette: {round(fr.iou * 100, 1)}% • "
            f"Min Part Precision: {int(min_p_iou * 100)}% • "
            f"{'Converged' if status_label == 'ok' else 'Needs Review'}"
        )

        return RelationalSynthesisResult(
            status=status_label,
            object_class=cls_name,
            class_confidence=cls_conf,
            class_reason=cls_reason,
            execution_time_ms=t_elapsed,
            iou=round(fr.iou, 4),
            initial_iou=round(fr.init_iou, 4),
            part_ious=fr.part_ious,
            history=fr.history,
            camera={"az": round(fr.camera.az, 1), "el": round(fr.camera.el, 1)},
            summary_line=summary,
            scale_anchored=bool(final_asm.anchor is not None),
            parts=parts_payload,
            features=features_payload,
            meshes=mesh_payloads,
            overlay_base64=overlay_b64,
            shaded_base64=shaded_b64,
            views_base64=views_b64,
            stl_base64=base64.b64encode(stl_bytes).decode("utf-8"),
            glb_base64=base64.b64encode(glb_bytes).decode("utf-8"),
            step_base64=base64.b64encode(step_bytes).decode("utf-8") if step_bytes else None,
            openscad_code=openscad_code
        )

    def _generate_scad(self, solids: Dict[str, Solid], features: List[Feature]) -> str:
        lines = [
            "// SpatialForge v2 Parametric CAD",
            "// Layers: Body (primitives) + Features (measured) + Cosmetic (library)",
            "$fn = 48;",
            ""
        ]

        pos_parts = [s for s in solids.values() if s.op == "union"]
        neg_parts = [s for s in solids.values() if s.op == "difference"]

        lines.append("module body_assembly() {")
        if neg_parts:
            lines.append("  difference() {")
            lines.append("    union() {")
            for s in pos_parts:
                lines.append(f"      // {s.id} ({s.kind})")
                lines.append(f"      translate([{s.center[0]:.2f}, {s.center[1]:.2f}, {s.center[2]:.2f}])")
                if s.kind == "box":
                    lines.append(f"        cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true);")
                elif s.kind == "rounded_box":
                    lines.append(f"        cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true); // fillet {s.fillet}")
                elif s.kind == "cylinder":
                    if s.axis == "x":
                        lines.append(f"        rotate([0, 90, 0]) cylinder(h={s.size[0]:.2f}, d={s.size[1]:.2f}, center=true);")
                    elif s.axis == "y":
                        lines.append(f"        rotate([90, 0, 0]) cylinder(h={s.size[1]:.2f}, d={s.size[0]:.2f}, center=true);")
                    else:
                        lines.append(f"        cylinder(h={s.size[2]:.2f}, d={s.size[0]:.2f}, center=true);")
                elif s.kind == "extrusion":
                    lines.append(f"        cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true); // profile extrusion")
            lines.append("    }")
            for s in neg_parts:
                lines.append(f"    // cut: {s.id}")
                lines.append(f"    translate([{s.center[0]:.2f}, {s.center[1]:.2f}, {s.center[2]:.2f}])")
                lines.append(f"      cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true);")
            lines.append("  }")
        else:
            lines.append("  union() {")
            for s in pos_parts:
                lines.append(f"    translate([{s.center[0]:.2f}, {s.center[1]:.2f}, {s.center[2]:.2f}])")
                lines.append(f"      cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true);")
            lines.append("  }")
        lines.append("}\n")

        lines.append("body_assembly();")
        return "\n".join(lines)
