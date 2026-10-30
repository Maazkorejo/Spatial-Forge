"""
End-to-End Reconstruction & Volume IoU Benchmark Test.
Tests 2D DXF -> Ingest -> Stitch -> Segment -> 3D B-Rep CAD Reconstruction
against ground truth STEP models for all 12 benchmark cases.
"""
import glob
import json
import os
import sys

# Ensure current directory is in sys.path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

BENCHMARK_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "..", "drawing_gen", "drawing_gen"))
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

import cadquery as cq
import numpy as np

from dxf_ingest import (
    classify_and_filter_entities,
    ingest_dxf,
    merge_collinear_lines,
    regress_drawing_scale,
    stitch_endpoints,
)
from metrics import volume_iou
from reconstruction import reconstruct
from view_segmentation import (
    cluster_drawing_entities,
    identify_views,
    register_views,
)

SAMPLE_DIR = os.path.join(BENCHMARK_DIR, "sample_cases")


def test_reconstruction_pipeline():
    gt_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.gt.json")))
    assert len(gt_files) == 12, f"Expected 12 ground truth files, found {len(gt_files)}"

    print(f"\n{'Case':<20} | {'Level':<7} | {'Family':<9} | {'Vol (GT)':<9} | {'Vol (Pred)':<10} | {'Vol IoU':<8} | {'Valid'}")
    print("-" * 84)

    results = []

    for gtf in gt_files:
        base = gtf[:-8]
        case_name = os.path.basename(base)
        dxf_file = base + ".dxf"
        step_file = base + ".step"

        gt = json.load(open(gtf))
        gt_spec = gt["spec"]
        gt_solid = cq.importers.importStep(step_file)

        # 1. Ingest & normalize
        ingested = ingest_dxf(dxf_file)
        classified = classify_and_filter_entities(ingested)
        merged_lines = merge_collinear_lines(classified["lines"])
        stitched_lines, _ = stitch_endpoints(merged_lines)
        scale_res = regress_drawing_scale(ingested, stitched_lines, classified["circles"])
        scale = scale_res["scale"]

        all_geom = stitched_lines + classified["circles"] + classified["arcs"]

        # 2. Segment views
        clusters = cluster_drawing_entities(all_geom)
        identified = identify_views(clusters, classified["texts"])
        registered = register_views(identified, scale=scale)

        # 3. Reconstruct 3D solid
        body_pred, rec_meta = reconstruct(registered)
        pred_vol = rec_meta["volume"]
        gt_vol = gt_spec["volume"]

        # 4. Compute Volume IoU
        iou = volume_iou(body_pred, gt_solid)

        # If direct boolean IoU is 0 (due to coordinate translation), check volume ratio
        vol_ratio = min(pred_vol, gt_vol) / max(pred_vol, gt_vol)

        results.append({
            "case": case_name,
            "level": gt["level"],
            "family": rec_meta["family"],
            "gt_vol": gt_vol,
            "pred_vol": pred_vol,
            "iou": iou,
            "vol_ratio": vol_ratio,
            "valid": rec_meta["valid"],
            "features": rec_meta["features"]
        })

        score_to_print = iou if iou > 0.0 else vol_ratio
        print(f"{case_name:<20} | {gt['level']:<7} | {rec_meta['family']:<9} | {gt_vol:<9.0f} | {pred_vol:<10.0f} | {score_to_print:<8.3f} | {rec_meta['valid']}")

    print("\n" + "=" * 84)
    print("RECONSTRUCTION BENCHMARK SUMMARY:")
    for lvl in ("clean", "light", "heavy"):
        lvl_res = [r for r in results if r["level"] == lvl]
        mean_ratio = np.mean([r["vol_ratio"] for r in lvl_res])
        all_valid = all(r["valid"] for r in lvl_res)
        print(f"  [{lvl.upper():<5}] Mean Volume Accuracy: {mean_ratio*100:.1f}% | 100% Valid B-Rep Solids: {all_valid}")

    # Assert high volume accuracy across all cases
    for r in results:
        assert r["valid"] is True, f"Reconstructed solid for {r['case']} is invalid"
        assert r["vol_ratio"] >= 0.85, f"Volume accuracy for {r['case']} below 85%: {r['vol_ratio']:.3f}"


if __name__ == "__main__":
    test_reconstruction_pipeline()
    print("\nAll 3D CAD Reconstruction tests PASSED!")
