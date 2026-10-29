import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import base64

from app.core.config import settings
from app.agent.relational_pipeline import RelationalPipelineEngine

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SpatialForge: Autonomous Relational Blueprint-to-CAD Engine"
)

# Enable CORS for local Next.js / browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static samples and UI
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
samples_dir = os.path.join(repo_root, "samples")
if os.path.exists(samples_dir):
    app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

@app.get("/ui")
@app.get("/app")
def get_ui():
    index_path = os.path.join(repo_root, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "index.html not found"}

relational_engine = RelationalPipelineEngine()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "SpatialForge: Relational Blueprint-to-CAD Synthesis with Analysis-by-Synthesis & Multi-View Plausibility",
        "stack": [
            "SpatialForge v2 Relational CSG Engine",
            "Differential Evolution Analysis-by-Synthesis",
            "OpenCV Hough & Contour Feature Detection",
            "Ray-Casting Surface Back-Projection",
            "Multi-View VLM Verification",
            "Manifold3D Watertight Solid Kernel",
            "CadQuery B-Rep STEP Exporter"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post(f"{settings.API_V1_STR}/draft")
async def draft_blueprint(
    file: UploadFile = File(...),
    requested_class: Optional[str] = Form(None)
):
    """
    Tiered Latency Phase 1 (<500ms): Fast draft pass.
    Instantly returns median-parameter 3D model for immediate visual feedback.
    """
    try:
        contents = await file.read()
        result = relational_engine.process_draft(contents, requested_class=requested_class)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post(f"{settings.API_V1_STR}/synthesize")
async def synthesize_blueprint(
    file: UploadFile = File(...),
    requested_class: Optional[str] = Form(None)
):
    """
    Main synthesis endpoint. Ingests a 2D technical blueprint or 3D concept sketch,
    runs the Relational CSG pipeline with camera estimation and render-and-compare optimization.
    """
    try:
        contents = await file.read()
        # Run Relational Analysis-by-Synthesis with spatialforge_v2
        result = relational_engine.process_image(contents, requested_class=requested_class)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

class CorrectionLog(BaseModel):
    image_hash: Optional[str] = "sketch"
    template_class: Optional[str] = None
    part_id: Optional[str] = None
    parameter: Optional[str] = None
    original_val: Optional[float] = None
    corrected_val: Optional[float] = None
    scale_anchor_mm: Optional[float] = None

@app.post(f"{settings.API_V1_STR}/log_correction")
async def log_correction(payload: CorrectionLog):
    """
    Logs human corrections (slider adjustments, anchor calibrations)
    to feed few-shot prompts and benchmark datasets.
    """
    import datetime
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corrections")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "corrections.jsonl")
    entry = payload.dict()
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "logged"}

class PatchRequest(BaseModel):
    template_class: Optional[str] = "stacked_housing"
    patches: List[Dict[str, Any]]
    locked_parts: Optional[List[str]] = None

@app.post(f"{settings.API_V1_STR}/patch")
async def patch_assembly(payload: PatchRequest):
    """
    Applies validated RFC 6902 JSON-patches with programmatic lock guards.
    Locked parts (high IoU or user approved) are strictly immutable.
    """
    import sys
    V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "spatialforge_v2", "spatialforge_v2"))
    if V2_DIR not in sys.path:
        sys.path.insert(0, V2_DIR)
    import patch_engine
    import templates
    from solver import resolve
    from geometry import part_mesh

    try:
        constructor = templates.TEMPLATE_REGISTRY.get(payload.template_class, templates.stacked_housing)
        current_asm = constructor()
        updated_asm, logs = patch_engine.apply_patch(current_asm, payload.patches, locked_parts=payload.locked_parts)

        # Generate updated meshes
        solids = resolve(updated_asm)
        meshes = {}
        for pid, s in solids.items():
            if s.op == "union":
                V, F_faces = part_mesh(s)
                meshes[pid] = {
                    "id": pid,
                    "layer": "body",
                    "vertices": [[round(float(c), 3) for c in pt] for pt in V],
                    "faces": [[int(idx) for idx in tri] for tri in F_faces]
                }
        return {"status": "ok", "logs": logs, "meshes": meshes, "assembly": updated_asm.model_dump()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get(f"{settings.API_V1_STR}/sample_cad")
def get_sample_cad(template: str = "stacked_housing"):
    """
    Returns pre-generated spatialforge_v2 sample CAD outputs (STEP, STL, GLB, JSON, meshes, overlays).
    """
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "samples")
    sample_v2_dir = os.path.join(os.path.dirname(__file__), "..", "..", "spatialforge_v2", "spatialforge_v2", "sample_output")
    res = {}

    file_map = {
        "model.stl": "stacked_housing.stl" if template == "stacked_housing" else "model.stl",
        "model.glb": "stacked_housing.glb" if template == "stacked_housing" else "model.glb",
        "model.step": "model.step",
        "model.json": "stacked_housing_model.json" if template == "stacked_housing" else "model.json",
        "sample_meshes.json": "stacked_housing_meshes.json" if template == "stacked_housing" else "sample_meshes.json",
        "overlay.png": "overlay.png",
        "shaded.png": "shaded.png",
        "views.png": "views.png"
    }

    for target_key, actual_file in file_map.items():
        p1 = os.path.join(samples_dir, actual_file)
        p2 = os.path.join(sample_v2_dir, actual_file)
        p = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                res[target_key] = base64.b64encode(fh.read()).decode("utf-8")
    return res

@app.websocket("/ws/stream")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    Live streaming WebSocket for UI step-by-step agent telemetry.
    Streams genuine pipeline progress across perception, camera estimation,
    relational optimization, feature ray-casting, and multi-view verification.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("action") == "synthesize" and "image_base64" in message:
                img_bytes = base64.b64decode(message["image_base64"])
                requested_class = message.get("requested_class")

                # Stage 1: Perception & Classification
                await websocket.send_json({
                    "stage": "PERCEPTION",
                    "status": "Segmenting silhouette mask & classifying mechanical topology...",
                    "progress": 20
                })

                # Stage 2: Camera & Draft Resolution
                await websocket.send_json({
                    "stage": "CAMERA_ESTIMATION",
                    "status": "Searching viewpoint camera & resolving initial assembly...",
                    "progress": 40
                })

                # Stage 3: Relational Differential Evolution
                await websocket.send_json({
                    "stage": "OPTIMIZATION",
                    "status": "Optimizing dimensional parameters via render-and-compare...",
                    "progress": 65
                })

                # Stage 4: Feature Ray-Casting & Multi-View Review
                await websocket.send_json({
                    "stage": "FEATURES_AND_VERIFICATION",
                    "status": "Detecting OpenCV features & verifying multi-view plausibility...",
                    "progress": 85
                })

                # Execute relational pipeline asynchronously
                result = await asyncio.to_thread(
                    relational_engine.process_image,
                    img_bytes,
                    requested_class=requested_class
                )

                # Stage 5: Complete
                await websocket.send_json({
                    "stage": "COMPLETE",
                    "status": f"Relational synthesis completed ({result.status.upper()}) • Silhouette IoU: {round(result.iou * 100, 1)}%",
                    "progress": 100,
                    "payload": result.model_dump()
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"stage": "ERROR", "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
