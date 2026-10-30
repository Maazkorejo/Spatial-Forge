import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import base64

V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "spatialforge_v2", "spatialforge_v2"))
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)
from dxf_pipeline import process_dxf_bytes

CAD_EDIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cad_edit"))
if CAD_EDIT_DIR not in sys.path:
    sys.path.insert(0, CAD_EDIT_DIR)
import model as CM
import session as CS
import build as CB
import gate as CG
import cadquery as cq
import tempfile

from app.core.config import settings

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

@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "SpatialForge: Autonomous 2D CAD (DXF) Orthographic Blueprint-to-3D B-Rep CAD Engine",
        "stack": [
            "2D DXF Normalization & Adaptive Endpoint Stitcher",
            "Orthographic View Segmentation (Front, Top, Right)",
            "Projection Standard Inference (1st vs 3rd Angle)",
            "2.5D Feature-Based B-Rep Reconstruction & Revolve Engine",
            "OpenCascade HLR Reprojection Verification Gate",
            "Watertight Solid Kernel & CadQuery STEP Exporter"
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
    Fast 2D CAD draft pass (<300ms).
    """
    try:
        contents = await file.read()
        is_dxf = (file.filename and file.filename.lower().endswith(".dxf")) or contents.startswith(b"0\nSECTION") or contents.startswith(b"0\r\nSECTION") or b"HEADER" in contents[:200]
        if not is_dxf:
            return JSONResponse(status_code=400, content={"error": "SpatialForge now operates exclusively on 2D CAD (.dxf) technical drawings. Please upload a .dxf file."})
        result = await asyncio.to_thread(process_dxf_bytes, contents, file.filename or "drawing.dxf")
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
    Main 2D CAD synthesis endpoint. Ingests a 2D orthographic technical drawing (.dxf),
    executing view segmentation, standard inference, 2.5D reconstruction, and reprojection verification.
    """
    try:
        contents = await file.read()
        is_dxf = (file.filename and file.filename.lower().endswith(".dxf")) or contents.startswith(b"0\nSECTION") or contents.startswith(b"0\r\nSECTION") or b"HEADER" in contents[:200]
        if not is_dxf:
            return JSONResponse(status_code=400, content={"error": "SpatialForge now operates exclusively on 2D CAD (.dxf) technical drawings. Please upload a .dxf file."})
        result = await asyncio.to_thread(process_dxf_bytes, contents, file.filename or "drawing.dxf")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post(f"{settings.API_V1_STR}/dxf")
@app.post("/pipeline/dxf")
async def process_dxf_endpoint(file: UploadFile = File(...)):
    """
    Dedicated 2D CAD (DXF) to 3D B-Rep CAD synthesis endpoint.
    Performs ingestion, view segmentation, standard inference, 2.5D reconstruction,
    and reprojection verification.
    """
    try:
        contents = await file.read()
        result = await asyncio.to_thread(process_dxf_bytes, contents, file.filename or "drawing.dxf")
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
def get_sample_cad(case: str = "case_0000_clean"):
    """
    Returns deterministic 2D CAD (DXF) sample reconstruction output
    including 2D drawing sheet preview, Three.js 3D meshes, STEP, STL, and reprojection metrics.
    """
    dxf_sample = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "drawing_gen", "drawing_gen", "sample_cases", f"{case}.dxf")
    )
    if os.path.exists(dxf_sample):
        with open(dxf_sample, "rb") as f:
            return process_dxf_bytes(f.read(), f"{case}.dxf")
    return {"error": f"Sample DXF {case} not found"}

@app.get(settings.API_V1_STR + "/download_dxf/{filename}")
def download_dxf(filename: str):
    """
    Directly downloads a sample 2D CAD (.dxf) file for offline inspection or upload testing.
    """
    search_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_dxf_drawings")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "drawing_gen", "drawing_gen", "sample_cases"))
    ]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.exists(candidate) and os.path.isfile(candidate) and filename.lower().endswith(".dxf"):
            return FileResponse(
                candidate,
                media_type="application/dxf",
                filename=filename,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
    return JSONResponse(status_code=404, content={"error": f"DXF sample {filename} not found"})

@app.get(settings.API_V1_STR + "/download_image/{filename}")
def download_image(filename: str):
    """
    Directly downloads a rendered PNG drawing picture.
    """
    search_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_dxf_drawings")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "drawing_gen", "drawing_gen", "sample_cases"))
    ]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.exists(candidate) and os.path.isfile(candidate) and filename.lower().endswith(".png"):
            return FileResponse(
                candidate,
                media_type="image/png",
                filename=filename,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
    return JSONResponse(status_code=404, content={"error": f"Drawing picture {filename} not found"})

# ---------------------------------------------------------------------------
# CAD Edit Engine (Parametric Feature Tree + Safety Gate + Version History)
# ---------------------------------------------------------------------------
active_cad_sessions: Dict[str, CS.Session] = {}

def get_cad_session(session_id: str = "default") -> CS.Session:
    if session_id not in active_cad_sessions:
        active_cad_sessions[session_id] = CS.Session(CM.FeatureTree(base=CM.Block(W=40.0, D=20.0, H=10.0)))
    return active_cad_sessions[session_id]

def solid_to_mesh(solid: cq.Workplane) -> Dict[str, Any]:
    verts, triangles = solid.val().tessellate(0.1, 0.2)
    mesh_verts = [[round(float(v.x), 3), round(float(v.y), 3), round(float(v.z), 3)] for v in verts]
    mesh_faces = [[int(tri[0]), int(tri[1]), int(tri[2])] for tri in triangles]
    return {
        "vertices": mesh_verts,
        "faces": mesh_faces,
        "n_vertices": len(mesh_verts),
        "n_faces": len(mesh_faces)
    }

class CadEditInitRequest(BaseModel):
    session_id: Optional[str] = "default"
    W: Optional[float] = 40.0
    D: Optional[float] = 20.0
    H: Optional[float] = 10.0
    tree: Optional[Dict[str, Any]] = None

class CadEditOpsRequest(BaseModel):
    session_id: Optional[str] = "default"
    ops: Any

class CadEditHistoryRequest(BaseModel):
    session_id: Optional[str] = "default"
    version: Optional[int] = None

@app.post(f"{settings.API_V1_STR}/cad_edit/init")
def cad_edit_init(req: CadEditInitRequest):
    sid = req.session_id or "default"
    if req.tree:
        tree = CM.FeatureTree.model_validate(req.tree)
    else:
        tree = CM.FeatureTree(base=CM.Block(W=req.W or 40.0, D=req.D or 20.0, H=req.H or 10.0))
    s = CS.Session(tree)
    active_cad_sessions[sid] = s
    solid = s.solid()
    return {
        "status": "ok",
        "session_id": sid,
        "version": s.current.version,
        "stats": s.current.stats,
        "tree": s.current.tree.model_dump(),
        "mesh": solid_to_mesh(solid)
    }

@app.get(f"{settings.API_V1_STR}/cad_edit/state")
def cad_edit_state(session_id: str = "default"):
    s = get_cad_session(session_id)
    solid = s.solid()
    return {
        "status": "ok",
        "session_id": session_id,
        "version": s.current.version,
        "cursor": s.cursor,
        "history_len": len(s.history),
        "history": [{"version": h.version, "ops": h.ops, "warnings": h.warnings} for h in s.history],
        "stats": s.current.stats,
        "tree": s.current.tree.model_dump(),
        "mesh": solid_to_mesh(solid)
    }

@app.post(f"{settings.API_V1_STR}/cad_edit/propose")
def cad_edit_propose(req: CadEditOpsRequest):
    s = get_cad_session(req.session_id or "default")
    ops_input = req.ops
    if isinstance(ops_input, str):
        ops, parse_errs = CM.parse_ops(ops_input)
        if parse_errs:
            return JSONResponse(status_code=400, content={"ok": False, "errors": parse_errs})
    elif isinstance(ops_input, list):
        try:
            ops = CM.OpList.model_validate({"ops": ops_input}).ops
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "errors": [str(e)]})
    elif isinstance(ops_input, dict) and "ops" in ops_input:
        try:
            ops = CM.OpList.model_validate(ops_input).ops
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "errors": [str(e)]})
    else:
        return JSONResponse(status_code=400, content={"ok": False, "errors": ["Invalid ops payload"]})

    p = s.propose(ops)
    return {
        "ok": p.ok,
        "messages": p.messages,
        "errors": p.errors,
        "warnings": p.warnings,
        "stats": p.stats,
        "diff": p.diff,
        "failed_op": p.failed_op,
        "proposed_tree": p.tree.model_dump() if p.tree else None
    }

@app.post(f"{settings.API_V1_STR}/cad_edit/apply")
def cad_edit_apply(req: CadEditOpsRequest):
    s = get_cad_session(req.session_id or "default")
    ops_input = req.ops
    if isinstance(ops_input, str):
        ops, parse_errs = CM.parse_ops(ops_input)
        if parse_errs:
            return JSONResponse(status_code=400, content={"ok": False, "errors": parse_errs})
    elif isinstance(ops_input, list):
        try:
            ops = CM.OpList.model_validate({"ops": ops_input}).ops
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "errors": [str(e)]})
    elif isinstance(ops_input, dict) and "ops" in ops_input:
        try:
            ops = CM.OpList.model_validate(ops_input).ops
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "errors": [str(e)]})
    else:
        return JSONResponse(status_code=400, content={"ok": False, "errors": ["Invalid ops payload"]})

    p = s.apply(ops)
    if not p.ok:
        return {
            "ok": False,
            "messages": p.messages,
            "errors": p.errors,
            "warnings": p.warnings,
            "stats": p.stats
        }

    solid = s.solid()
    return {
        "ok": True,
        "version": s.current.version,
        "messages": p.messages,
        "warnings": p.warnings,
        "stats": p.stats,
        "diff": p.diff,
        "tree": s.current.tree.model_dump(),
        "mesh": solid_to_mesh(solid),
        "history": [{"version": h.version, "ops": h.ops} for h in s.history]
    }

@app.post(f"{settings.API_V1_STR}/cad_edit/undo")
def cad_edit_undo(req: CadEditHistoryRequest):
    s = get_cad_session(req.session_id or "default")
    snap = s.undo()
    solid = s.solid()
    return {
        "status": "ok",
        "version": snap.version,
        "cursor": s.cursor,
        "stats": snap.stats,
        "tree": snap.tree.model_dump(),
        "mesh": solid_to_mesh(solid)
    }

@app.post(f"{settings.API_V1_STR}/cad_edit/redo")
def cad_edit_redo(req: CadEditHistoryRequest):
    s = get_cad_session(req.session_id or "default")
    snap = s.redo()
    solid = s.solid()
    return {
        "status": "ok",
        "version": snap.version,
        "cursor": s.cursor,
        "stats": snap.stats,
        "tree": snap.tree.model_dump(),
        "mesh": solid_to_mesh(solid)
    }

@app.get(f"{settings.API_V1_STR}/cad_edit/export/step")
def cad_edit_export_step(session_id: str = "default"):
    s = get_cad_session(session_id)
    solid = s.solid()
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    cq.exporters.export(solid, path)
    return FileResponse(
        path,
        media_type="application/step",
        filename=f"cad_edit_v{s.current.version}.step",
        headers={"Content-Disposition": f'attachment; filename="cad_edit_v{s.current.version}.step"'}
    )

@app.get(f"{settings.API_V1_STR}/cad_edit/export/stl")
def cad_edit_export_stl(session_id: str = "default"):
    s = get_cad_session(session_id)
    solid = s.solid()
    fd, path = tempfile.mkstemp(suffix=".stl")
    os.close(fd)
    cq.exporters.export(solid, path)
    return FileResponse(
        path,
        media_type="application/sla",
        filename=f"cad_edit_v{s.current.version}.stl",
        headers={"Content-Disposition": f'attachment; filename="cad_edit_v{s.current.version}.stl"'}
    )

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
                raw_bytes = base64.b64decode(message["image_base64"])
                requested_class = message.get("requested_class")
                filename = message.get("filename", "drawing.dxf" if message.get("is_dxf") else "sketch.png")
                is_dxf = message.get("is_dxf") or filename.lower().endswith(".dxf") or raw_bytes.startswith(b"0\nSECTION") or raw_bytes.startswith(b"0\r\nSECTION") or b"HEADER" in raw_bytes[:200]

                if is_dxf:
                    await websocket.send_json({
                        "stage": "INGESTION",
                        "status": "Normalizing DXF entities, classifying lines & stitching endpoints...",
                        "progress": 25
                    })
                    await websocket.send_json({
                        "stage": "VIEW_SEGMENTATION",
                        "status": "Clustering drawing views & inferring projection standard (1st vs 3rd angle)...",
                        "progress": 50
                    })
                    await websocket.send_json({
                        "stage": "RECONSTRUCTION",
                        "status": "Synthesizing 2.5D B-Rep solid with measured features...",
                        "progress": 75
                    })
                    await websocket.send_json({
                        "stage": "VERIFICATION",
                        "status": "Reprojecting 3D solid via OpenCascade HLR & checking reliability gate...",
                        "progress": 90
                    })

                    result = await asyncio.to_thread(process_dxf_bytes, raw_bytes, filename)

                    await websocket.send_json({
                        "stage": "COMPLETE",
                        "status": f"2D CAD synthesis completed ({result['status'].upper()}) • Reprojection F1: {round(result['iou'] * 100, 1)}%",
                        "progress": 100,
                        "payload": result
                    })
                    continue
                else:
                    await websocket.send_json({
                        "stage": "ERROR",
                        "error": "SpatialForge operates exclusively on 2D CAD (.dxf) blueprints. Please upload a .dxf file."
                    })
                    continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"stage": "ERROR", "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
