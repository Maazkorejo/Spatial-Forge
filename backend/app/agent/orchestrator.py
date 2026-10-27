import time
import base64
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.perception.contour_extractor import BlueprintPerceptionEngine, PartTopology
from app.ocr.micro_ocr import MicroOCREngine, DimensionCallout
from app.agent.drc_auditor import DRCAuditor, DRCAuditReport
from app.cad.openscad_generator import ParametricCADGenerator
from app.verification.self_healer import SelfHealingLoop, SelfHealingIterationTrace
from app.core.config import settings

class SynthesisResponse(BaseModel):
    execution_time_ms: float
    topology: PartTopology
    dimension_callouts: List[DimensionCallout]
    drc_report: DRCAuditReport
    openscad_code: str
    stl_base64: str
    self_healing_traces: List[SelfHealingIterationTrace]
    final_parameters: Dict[str, float]

class PipelineOrchestrator:
    """
    Central Coordinator for the SpatialForge Agentic Pipeline.
    Connects Perception -> Micro-OCR -> DRC Audit -> Parametric CAD -> Self-Healing.
    """

    def __init__(self):
        self.perception_engine = BlueprintPerceptionEngine()
        self.ocr_engine = MicroOCREngine()
        self.drc_auditor = DRCAuditor(min_wall_thickness=settings.MIN_WALL_THICKNESS_MM)
        self.cad_generator = ParametricCADGenerator()
        self.healer = SelfHealingLoop(target_iou=settings.IOU_CONVERGENCE_THRESHOLD)

    async def execute_pipeline(self, image_bytes: bytes) -> SynthesisResponse:
        t0 = time.perf_counter()

        # Step 1: Perception (Contour & Hull Extraction)
        topology, binary_img = self.perception_engine.process_image(image_bytes)

        # Step 2: Micro-OCR on cropped dimension annotations
        callouts = self.ocr_engine.extract_dimensions_from_crops(binary_img, topology.annotation_regions)

        # Determine target dimensions (using OCR if detected, otherwise topology estimate)
        target_len = topology.estimated_length_mm
        target_wid = topology.estimated_width_mm
        target_bore = 44.0  # default mechanical bore

        # Match OCR results
        for c in callouts:
            if c.symbol_type == "diameter" and c.numeric_value > 5.0:
                target_bore = c.numeric_value
            elif c.symbol_type == "linear":
                if c.numeric_value > target_len * 0.7:
                    target_len = c.numeric_value
                else:
                    target_wid = c.numeric_value

        # Step 3: Manufacturing DRC Audit
        drc = self.drc_auditor.audit(
            length=target_len,
            width=target_wid,
            thickness=settings.DEFAULT_PLATE_THICKNESS_MM,
            bore_dia=target_bore
        )

        # Step 4: Parametric CAD Synthesis
        scad_code = self.cad_generator.generate_openscad_code(
            length=target_len,
            width=target_wid,
            thickness=settings.DEFAULT_PLATE_THICKNESS_MM,
            bore_dia=target_bore
        )

        mesh, stl_bytes = self.cad_generator.compile_mesh_in_memory(
            length=target_len,
            width=target_wid,
            thickness=settings.DEFAULT_PLATE_THICKNESS_MM,
            bore_dia=target_bore
        )

        # Step 5: Self-Healing Verification Loop
        converged, iou, healed_params, traces = self.healer.verify_and_heal(
            mesh=mesh,
            target_length=target_len,
            target_width=target_wid,
            target_bore=target_bore,
            current_length=target_len,
            current_width=target_wid,
            current_bore=target_bore
        )

        t_elapsed = round((time.perf_counter() - t0) * 1000.0, 1)

        return SynthesisResponse(
            execution_time_ms=t_elapsed,
            topology=topology,
            dimension_callouts=callouts,
            drc_report=drc,
            openscad_code=scad_code,
            stl_base64=base64.b64encode(stl_bytes).decode('utf-8'),
            self_healing_traces=traces,
            final_parameters=healed_params
        )
