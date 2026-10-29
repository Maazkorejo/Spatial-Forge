"""
SpatialForge 2D Benchmark Runner & Baseline Tester
Wires the `drawing_gen` sample benchmark cases into the SpatialForge test suite.
Evaluates DXF ingestion, entity classification, dimension consistency, and edge/volume metrics.
"""
import os
import sys
import json
import math
import glob
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

# Add drawing_gen directory to path
BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawing_gen", "drawing_gen")
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from dxf_io import read_dxf


def sample_primitive(p: dict, step: float = 0.5) -> np.ndarray:
    """Points along a primitive, spacing <= step."""
    kind = p.get("kind") or p.get("t")
    g = p.get("g", p)
    if kind == "line":
        x0, y0, x1, y1 = g["p"]
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        s = np.linspace(0, 1, n)
        return np.c_[x0 + (x1 - x0) * s, y0 + (y1 - y0) * s]
    if kind in ("circle", "arc"):
        cx, cy = g["c"]
        r = g["r"]
        if kind == "circle":
            a = np.linspace(0, 2 * math.pi, max(8, int(2 * math.pi * r / step)), endpoint=False)
        else:
            a0 = math.radians(g["a0"])
            sweep = (math.radians(g["a1"]) - a0) % (2 * math.pi)
            a = a0 + np.linspace(0, sweep, max(3, int(sweep * r / step) + 1))
        return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    if "pts" in g:
        return np.asarray(g["pts"], float)
    return np.zeros((0, 2))


def get_edge_points(ents: list, roles=("visible",), step: float = 0.5) -> np.ndarray:
    pts = [sample_primitive(e, step) for e in ents if e.get("role", "visible") in roles and (e.get("kind") or e.get("t")) in ("line", "circle", "arc")]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def compute_edge_prf(pred_ents: list, gt_ents: list, roles=("visible",), tol: float = 0.6, step: float = 0.5) -> dict:
    P = get_edge_points(pred_ents, roles, step)
    G = get_edge_points(gt_ents, roles, step)
    if len(P) == 0 or len(G) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prec = float((cKDTree(G).query(P)[0] <= tol).mean())
    rec = float((cKDTree(P).query(G)[0] <= tol).mean())
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_case(case_name: str, dxf_path: str, gt_path: str, step_path: str | None = None) -> dict:
    """Evaluate a single benchmark case."""
    gt = json.load(open(gt_path, "r"))
    parsed_dxf = read_dxf(dxf_path)
    
    # 1. Entity Conversion: convert flat parsed_dxf to entity list
    raw_ents = []
    for l in parsed_dxf["lines"]:
        raw_ents.append({"kind": "line", "g": {"p": l["p"]}, "layer": l["layer"], "ltype": l["ltype"], "role": "unknown"})
    for c in parsed_dxf["circles"]:
        raw_ents.append({"kind": "circle", "g": {"c": c["c"], "r": c["r"]}, "layer": c["layer"], "ltype": c["ltype"], "role": "unknown"})
    for a in parsed_dxf["arcs"]:
        raw_ents.append({"kind": "arc", "g": {"c": a["c"], "r": a["r"], "a0": a["a0"], "a1": a["a1"]}, "layer": a["layer"], "ltype": a["ltype"], "role": "unknown"})
    
    clean_ents = gt["clean_entities"]
    
    # 2. Edge PRF: Visible edges
    # For clean/light, we can filter by layer/ltype; for heavy, test overall coverage
    vis_gt = [e for e in clean_ents if e.get("role") == "visible"]
    vis_pred = [e for e in raw_ents if e.get("layer") in ("VISIBLE", "0") and e.get("ltype") in ("CONTINUOUS", "BYLAYER")]
    
    prf_vis = compute_edge_prf(vis_pred, vis_gt, roles=("visible", "unknown"))
    
    # 3. Dimension cross-check
    dim_records = []
    scale = gt["meta"].get("scale", 1.0)
    for d in parsed_dxf["dims"]:
        meas = d["measurement"]
        text = d["text"]
        dim_records.append({"measurement": meas, "text": text, "ratio_to_scale": meas / scale if scale > 0 else 0})
        
    # 4. Check endpoint matching (for loop closure)
    lines_p = [e["g"]["p"] for e in raw_ents if e["kind"] == "line"]
    if lines_p:
        pts = [tuple(np.round(l[i:i + 2], 3)) for l in lines_p for i in (0, 2)]
        c = Counter(pts)
        endpoint_match_rate = sum(1 for p in pts if c[p] > 1) / len(pts)
    else:
        endpoint_match_rate = 0.0
        
    return {
        "case": case_name,
        "seed": gt["seed"],
        "level": gt["level"],
        "family": gt["spec"]["family"],
        "volume_gt": gt["spec"]["volume"],
        "raw_counts": {k: len(v) if isinstance(v, list) else v for k, v in parsed_dxf.items()},
        "vis_edge_f1": prf_vis["f1"],
        "vis_edge_prec": prf_vis["precision"],
        "vis_edge_rec": prf_vis["recall"],
        "endpoint_match_rate": endpoint_match_rate,
        "dims_found": len(dim_records),
        "noise_log": gt.get("noise_log", {})
    }


def run_benchmark():
    sample_dir = os.path.join(BENCHMARK_DIR, "sample_cases")
    gt_files = sorted(glob.glob(os.path.join(sample_dir, "*.gt.json")))
    print(f"Found {len(gt_files)} ground truth benchmark cases in {sample_dir}\n")
    
    results = []
    print(f"{'Case':<20} | {'Level':<7} | {'Family':<9} | {'Edge Prec':<9} | {'Edge Rec':<9} | {'Edge F1':<8} | {'Ep Match':<8} | {'Dims'}")
    print("-" * 88)
    
    for gtf in gt_files:
        base = gtf[:-8] # remove .gt.json
        case_name = os.path.basename(base)
        dxf_file = base + ".dxf"
        step_file = base + ".step"
        res = evaluate_case(case_name, dxf_file, gtf, step_file)
        results.append(res)
        print(f"{res['case']:<20} | {res['level']:<7} | {res['family']:<9} | {res['vis_edge_prec']:<9.3f} | {res['vis_edge_rec']:<9.3f} | {res['vis_edge_f1']:<8.3f} | {res['endpoint_match_rate']:<8.2f} | {res['dims_found']}")

    print("\n" + "=" * 88)
    print("BENCHMARK BASELINE SUMMARY:")
    for lvl in ("clean", "light", "heavy"):
        lvl_res = [r for r in results if r["level"] == lvl]
        if lvl_res:
            mean_f1 = np.mean([r["vis_edge_f1"] for r in lvl_res])
            mean_rec = np.mean([r["vis_edge_rec"] for r in lvl_res])
            mean_prec = np.mean([r["vis_edge_prec"] for r in lvl_res])
            mean_ep = np.mean([r["endpoint_match_rate"] for r in lvl_res])
            print(f"  [{lvl.upper():<5}] Mean Edge F1: {mean_f1:.3f} | Prec: {mean_prec:.3f} | Recall: {mean_rec:.3f} | Endpoint Match: {mean_ep:.2f}")

    return results


if __name__ == "__main__":
    run_benchmark()
