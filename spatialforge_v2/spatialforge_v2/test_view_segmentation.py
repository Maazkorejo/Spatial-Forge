"""
Unit and integration tests for View Segmentation, Standard Inference, and View Registration.
Validates 100% accuracy on first-angle vs third-angle detection and [W, D, H] recovery.
"""
import glob
import json
import os
import sys

# Ensure current directory is in sys.path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from dxf_ingest import (
    classify_and_filter_entities,
    ingest_dxf,
    merge_collinear_lines,
    regress_drawing_scale,
    stitch_endpoints,
)
from view_segmentation import (
    cluster_drawing_entities,
    identify_views,
    register_views,
)

SAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "drawing_gen", "drawing_gen", "sample_cases")
)


def test_view_segmentation_and_standard_inference():
    """Verify view clustering, front/top/right identification, and 1st/3rd angle detection."""
    gt_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.gt.json")))
    assert len(gt_files) == 12, f"Expected 12 ground truth files, found {len(gt_files)}"

    print(f"\n{'Case':<20} | {'Expected Std':<13} | {'Inferred Std':<13} | {'W x D x H (GT)':<18} | {'W x D x H (Pred)'}")
    print("-" * 88)

    passed_std = 0
    passed_dims = 0

    for gtf in gt_files:
        base = gtf[:-8]
        case_name = os.path.basename(base)
        dxf_file = base + ".dxf"
        gt = json.load(open(gtf))
        gt_std = gt["meta"]["standard"]
        gt_spec = gt["spec"]
        gt_dim_str = f"{gt_spec.get('W', 0)} x {gt_spec.get('D', 0)} x {gt_spec.get('H', 0)}"

        # 1. Ingest and normalize
        ingested = ingest_dxf(dxf_file)
        classified = classify_and_filter_entities(ingested)
        merged_lines = merge_collinear_lines(classified["lines"])
        stitched_lines, _ = stitch_endpoints(merged_lines)
        scale_res = regress_drawing_scale(ingested, stitched_lines, classified["circles"])
        scale = scale_res["scale"]

        all_geom = stitched_lines + classified["circles"] + classified["arcs"]

        # 2. Cluster into views
        clusters = cluster_drawing_entities(all_geom)
        assert len(clusters) >= 3, f"Failed to find >= 3 clusters for {case_name}, found {len(clusters)}"

        # 3. Identify views and standard
        identified = identify_views(clusters, classified["texts"])
        inferred_std = identified["standard"]

        if inferred_std == gt_std:
            passed_std += 1

        # 4. Register views
        registered = register_views(identified, scale=scale)
        dims = registered["dimensions"]
        pred_dim_str = f"{dims['W']} x {dims['D']} x {dims['H']}"

        # Allow small tolerance on dimensions
        if "W" in gt_spec:
            w_ok = abs(dims["W"] - gt_spec["W"]) <= 1.0
            d_ok = abs(dims["D"] - gt_spec["D"]) <= 1.0
            h_ok = abs(dims["H"] - gt_spec["H"]) <= 1.0
            if w_ok and d_ok and h_ok:
                passed_dims += 1

        print(f"{case_name:<20} | {gt_std:<13} | {inferred_std:<13} | {gt_dim_str:<18} | {pred_dim_str}")

    print("\n" + "=" * 88)
    print(f"Standard Detection Accuracy: {passed_std}/12 ({passed_std/12*100:.1f}%)")
    print(f"Dimension [W, D, H] Accuracy: {passed_dims}/12 ({passed_dims/12*100:.1f}%)")
    assert passed_std == 12, f"Expected 12/12 standard detections, got {passed_std}"
    assert passed_dims == 12, f"Expected 12/12 dimension recoveries, got {passed_dims}"


if __name__ == "__main__":
    test_view_segmentation_and_standard_inference()
    print("\nAll View Segmentation & Registration tests PASSED!")
