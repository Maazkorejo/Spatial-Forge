import sys
import os
import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import trimesh

# Include files directory
FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "files"))
if FILES_DIR not in sys.path:
    sys.path.insert(0, FILES_DIR)

import fit as F
from schema import Assembly, Part
from solver import resolve, geometry_report, Solid
from pipeline import extract_mask, encode_png
from examples import sewing_machine
from vlm import OpenAICompatVLM, OllamaVLM
from app.core.config import settings

class SolidPayload(BaseModel):
    id: str
    kind: str
    op: str
    cuts: List[str]
    center: List[float]
    size: List[float]
    axis: Optional[str] = None
    source: str = "inferred"

class RelationalSynthesisResult(BaseModel):
    status: str
    object_class: str
    execution_time_ms: float
    iou: float
    initial_iou: float
    camera: Dict[str, float]  # az, el
    solids: List[SolidPayload]
    parts: List[Dict[str, Any]]
    overlay_base64: Optional[str] = None
    openscad_code: str
    stl_base64: str

class RelationalPipelineEngine:
    """
    Bridge connecting the SpatialForge v2 Relational CSG pipeline
    with the FastAPI backend and Three.js frontend.
    """

    def __init__(self):
        self.vlm = None
        if settings.GROQ_API_KEY:
            self.vlm = OpenAICompatVLM(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
                model="llama-3.2-11b-vision-preview"
            )

    def process_image(self, img_bytes: bytes) -> RelationalSynthesisResult:
        import time
        t0 = time.perf_counter()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image bytes")

        # 1. Extract Target Mask
        mask = extract_mask(img_bgr)
        target = F.Target(mask, S=160)

        # 2. Candidate Proposal
        # If VLM is available, call it; otherwise use reference mechanical priors
        asm = None
        if self.vlm is not None:
            try:
                from pipeline import propose
                from prompt import build_system_prompt, build_user_prompt
                png = encode_png(img_bgr)
                asm = propose(self.vlm, [png], build_user_prompt(), build_system_prompt(), 0.2)
            except Exception as e:
                print(f"[Pipeline] VLM call failed or unavailable ({e}), falling back to structural prior.")

        if asm is None:
            asm = sewing_machine()

        # 3. Camera Search & Continuous Optimization
        cam = F.search_camera(asm, target)
        fit_res = F.fit(asm, target, cam=cam)
        fitted_asm = F.apply_fit(asm, fit_res)

        # 4. Resolve to World-Space Solids
        solids_dict = resolve(fitted_asm)

        # 5. Build Meshes & OpenSCAD
        scad_code, stl_bytes = self._generate_cad_and_stl(solids_dict)

        # 6. Generate Diagnostic Overlay
        overlay_b64 = None
        if fit_res.rendered is not None:
            overlay_bytes = F.make_overlay(fit_res.rendered, target)
            overlay_b64 = base64.b64encode(overlay_bytes).decode('utf-8')

        # 7. Format Solids Payload for Three.js
        solids_list = []
        for s in solids_dict.values():
            # Determine source label
            part_match = next((p for p in fitted_asm.parts if p.id == s.id), None)
            src = "inferred"
            if part_match:
                # If any dimension is assumed, mark accordingly
                dims = part_match.dims()
                if any(d.source == "assumed" for d in dims.values()):
                    src = "assumed"
                elif all(d.source == "measured" for d in dims.values()):
                    src = "measured"

            solids_list.append(SolidPayload(
                id=s.id,
                kind=s.kind,
                op=s.op,
                cuts=s.cuts,
                center=[round(float(c), 2) for c in s.center],
                size=[round(float(sz), 2) for sz in s.size],
                axis=s.axis,
                source=src
            ))

        # Format Parts with fitted dimensions
        parts_data = []
        for p in fitted_asm.parts:
            dims_dict = {f: {"value": round(d.value, 2), "source": d.source, "min": d.min, "max": d.max}
                         for f, d in p.dims().items()}
            parts_data.append({
                "id": p.id,
                "label": p.label,
                "kind": p.kind,
                "op": p.op,
                "dims": dims_dict
            })

        t_elapsed = round((time.perf_counter() - t0) * 1000.0, 1)

        return RelationalSynthesisResult(
            status="ok" if fit_res.iou >= 0.85 else "needs_review",
            object_class=fitted_asm.object_class,
            execution_time_ms=t_elapsed,
            iou=round(fit_res.iou, 4),
            initial_iou=round(fit_res.init_iou, 4),
            camera={"az": round(fit_res.camera.az, 1), "el": round(fit_res.camera.el, 1)},
            solids=solids_list,
            parts=parts_data,
            overlay_base64=overlay_b64,
            openscad_code=scad_code,
            stl_base64=base64.b64encode(stl_bytes).decode('utf-8')
        )

    def _generate_cad_and_stl(self, solids: Dict[str, Solid]) -> tuple[str, bytes]:
        # Generate OpenSCAD
        scad_lines = [
            "// SpatialForge Relational CSG Synthesis",
            "// Fitted via Analysis-by-Synthesis & scipy Optimizer",
            "$fn = 48;",
            ""
        ]

        pos_parts = [s for s in solids.values() if s.op == "union"]
        neg_parts = [s for s in solids.values() if s.op == "difference"]

        scad_lines.append("difference() {")
        scad_lines.append("  union() {")
        for s in pos_parts:
            scad_lines.append(f"    // {s.id} ({s.kind})")
            scad_lines.append(f"    translate([{s.center[0]:.2f}, {s.center[1]:.2f}, {s.center[2]:.2f}])")
            if s.kind == "box":
                scad_lines.append(f"      cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true);")
            elif s.kind == "cylinder":
                # handle cylinder axis
                scad_lines.append(f"      cylinder(h={s.size[2]:.2f}, d={s.size[0]:.2f}, center=true);")
            elif s.kind == "sphere":
                scad_lines.append(f"      sphere(d={s.size[0]:.2f});")
        scad_lines.append("  }")

        for s in neg_parts:
            scad_lines.append(f"    // cut: {s.id}")
            scad_lines.append(f"    translate([{s.center[0]:.2f}, {s.center[1]:.2f}, {s.center[2]:.2f}])")
            if s.kind == "box":
                scad_lines.append(f"      cube([{s.size[0]:.2f}, {s.size[1]:.2f}, {s.size[2]:.2f}], center=true);")
            elif s.kind == "cylinder":
                scad_lines.append(f"      cylinder(h={s.size[2]:.2f}, d={s.size[0]:.2f}, center=true);")
        scad_lines.append("}")

        scad_code = "\n".join(scad_lines)

        # Build in-memory Trimesh
        meshes = []
        for s in pos_parts:
            if s.kind == "box":
                m = trimesh.creation.box(extents=s.size)
                m.apply_translation(s.center)
                meshes.append(m)
            elif s.kind == "cylinder":
                # find length and diameter
                d = min(s.size[0], s.size[1])
                h = max(s.size)
                m = trimesh.creation.cylinder(radius=d/2.0, height=h, sections=32)
                m.apply_translation(s.center)
                meshes.append(m)

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            stl_data = combined.export(file_type='stl')
        else:
            stl_data = b""

        return scad_code, stl_data
