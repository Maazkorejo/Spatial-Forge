"""
Unit and integration tests for Reprojection Verification & Self-Flagging Gate.
Tests OpenCascade HLR reprojection against drawn DXF views and computes flag_stats.
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
from metrics import flag_stats, volume_iou
from reconstruction import reconstruct
from reproject_verify import verify_reconstruction
from view_segmentation import (
    cluster_drawing_entities,
    identify_views,
    register_views,
)

SAMPLE_DIR = os.path.join(BENCHMARK_DIR, "sample_cases")


def test_reprojection_verification():
    gt_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.gt.json")))
    assert len(gt_files) == 12, f"Expected 12 ground truth files, found {len(gt_files)}"

    print(f"\n{'Case':<20} | {'Level':<7} | {'Vol IoU':<8} | {'Reproj F1':<10} | {'Status':<13} | {'Flagged'}")
    print("-" * 75)

    records = []

    for gtf in gt_files:
        base = gtf[:-8]
        case_name = os.path.basename(base)
        dxf_file = base + ".dxf"
        step_file = base + ".step"

        gt = json.load(open(gtf))
        gt_solid = cq.importers.importStep(step_file)

        # 1. Ingest, stitch, segment
        ingested = ingest_dxf(dxf_file)
        classified = classify_and_filter_entities(ingested)
        merged_lines = merge_collinear_lines(classified["lines"])
        stitched_lines, _ = stitch_endpoints(merged_lines)
        scale_res = regress_drawing_scale(ingested, stitched_lines, classified["circles"])
        scale = scale_res["scale"]

        all_geom = stitched_lines + classified["circles"] + classified["arcs"]
        clusters = cluster_drawing_entities(all_geom)
        identified = identify_views(clusters, classified["texts"])
        registered = register_views(identified, scale=scale)

        # 2. Reconstruct
        body_pred, rec_meta = reconstruct(registered)
        iou = volume_iou(body_pred, gt_solid)

        # 3. Reprojection Verification
        verif = verify_reconstruction(body_pred, registered)
        records.append((iou, verif["flagged"]))

        print(f"{case_name:<20} | {gt['level']:<7} | {iou:<8.3f} | {verif['mean_vis_f1']:<10.3f} | {verif['status']:<13} | {verif['flagged']}")

    # 4. Test a deliberately corrupted solid to verify gate catches bad geometry
    print("\nTesting deliberate failure rejection...")
    bad_solid = cq.Workplane("XY").box(15, 15, 10, centered=False)  # tiny wrong box
    case_0_reg = register_views(identify_views(cluster_drawing_entities(all_geom)))
    bad_verif = verify_reconstruction(bad_solid, case_0_reg)
    bad_iou = volume_iou(bad_solid, gt_solid)
    records.append((bad_iou, bad_verif["flagged"]))
    print(f"Bad Solid Test: IoU={bad_iou:.3f}, Reproj F1={bad_verif['mean_vis_f1']:.3f}, Status={bad_verif['status']}, Flagged={bad_verif['flagged']}")
    assert bad_verif["flagged"] is True, "Reprojection gate failed to flag bad solid"

    # 5. Evaluate Self-Flagging Statistics
    print("\n" + "=" * 75)
    # Check Clean and Light noise tiers (normal / realistic drawings)
    clean_light_records = [records[i] for i, gtf in enumerate(gt_files) if json.load(open(gtf))["level"] in ("clean", "light")]
    clean_light_stats = flag_stats(clean_light_records, wrong_below=0.90)
    print("CLEAN & LIGHT NOISE SELF-FLAGGING:")
    print(f"  Cases: {clean_light_stats['n']}")
    print(f"  False Alarm Rate: {clean_light_stats['false_alarm']:.2f}")
    print(f"  Silent Failures: {clean_light_stats['silent_failures']}")
    assert clean_light_stats["silent_failures"] == 0, "Silent failure in clean/light cases"
    assert clean_light_stats["false_alarm"] == 0.0, f"False alarm in clean/light cases: {clean_light_stats['false_alarm']}"

    # Overall statistics (including heavy noise and deliberate bad solids)
    stats = flag_stats(records, wrong_below=0.90)
    print("\nOVERALL BENCHMARK SELF-FLAGGING REPORT:")
    print(f"  Total Cases Evaluated: {stats['n']}")
    print(f"  Wrong Reconstructions: {stats['n_wrong']}")
    print(f"  Right Reconstructions: {stats['n_right']}")
    print(f"  Flag Recall P(flagged | wrong): {stats['flag_recall']:.2f}")
    print(f"  Silent Failures: {stats['silent_failures']}")
    print(f"  Heavy-Noise Flags: {sum(1 for i, gtf in enumerate(gt_files) if json.load(open(gtf))['level'] == 'heavy' and records[i][1])}/4 (Dirty files safely routed to review)")

    assert stats["silent_failures"] == 0, f"Expected 0 silent failures, got {stats['silent_failures']}"
    assert stats["flag_recall"] == 1.0, f"Failed to catch wrong reconstructions: {stats['flag_recall']}"


if __name__ == "__main__":
    test_reprojection_verification()
    print("\nAll Reprojection Verification tests PASSED!")
