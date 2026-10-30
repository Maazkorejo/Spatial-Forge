"""
Unit and integration tests for DXF Ingestion, Normalization, Classification, Stitching, and Loop Closure.
Tests against benchmark sample cases across clean, light, and heavy noise tiers.
"""
import glob
import json
import os
import sys

# Ensure current directory is in sys.path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import numpy as np

from dxf_ingest import (
    build_closed_loops,
    classify_and_filter_entities,
    ingest_dxf,
    merge_collinear_lines,
    regress_drawing_scale,
    stitch_endpoints,
)

SAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "drawing_gen", "drawing_gen", "sample_cases")
)


def test_dxf_ingest_clean_structure():
    """Verify clean DXF reading extracts exact entity counts and units."""
    path = os.path.join(SAMPLE_DIR, "case_0000_clean.dxf")
    assert os.path.exists(path), f"File {path} does not exist"
    
    ingested = ingest_dxf(path)
    assert ingested["insunits"] == 4
    assert ingested["units"] == "mm"
    assert len(ingested["lines"]) > 0
    assert len(ingested["circles"]) == 3
    assert len(ingested["dims"]) == 5


def test_collinear_line_merging():
    """Verify collinear fragmented segments are merged into single continuous lines."""
    fragmented = [
        {"p": [0.0, 0.0, 10.0, 0.0], "layer": "VISIBLE", "ltype": "CONTINUOUS"},
        {"p": [10.0, 0.0, 20.0, 0.0], "layer": "VISIBLE", "ltype": "CONTINUOUS"},
        {"p": [20.0, 0.0, 30.0, 0.0], "layer": "VISIBLE", "ltype": "CONTINUOUS"},
        {"p": [0.0, 5.0, 0.0, 15.0], "layer": "VISIBLE", "ltype": "CONTINUOUS"},
    ]
    merged = merge_collinear_lines(fragmented, dist_tol=0.2)
    assert len(merged) == 2
    # Check that horizontal line was merged from 0 to 30
    h_lines = [l for l in merged if abs(l["p"][1] - l["p"][3]) < 1e-3]
    assert len(h_lines) == 1
    p = h_lines[0]["p"]
    assert abs(p[0] - 0.0) < 1e-3 and abs(p[2] - 30.0) < 1e-3


def test_adaptive_endpoint_stitching():
    """Verify gaps introduced at endpoints are closed by adaptive clustering."""
    # A rectangle with intentional 0.2mm gaps at each corner
    gapped_rect = [
        {"p": [0.2, 0.0, 19.8, 0.0], "role": "visible"},
        {"p": [20.0, 0.2, 20.0, 9.8], "role": "visible"},
        {"p": [19.8, 10.0, 0.2, 10.0], "role": "visible"},
        {"p": [0.0, 9.8, 0.0, 0.2], "role": "visible"},
    ]
    stitched, stats = stitch_endpoints(gapped_rect, user_tolerance=0.4)
    assert stats["gaps_stitched"] >= 4
    assert stats["max_gap_closed"] >= 0.15
    
    # After stitching, loops must form cleanly
    polys, _ = build_closed_loops(stitched)
    assert len(polys) == 1
    assert 190.0 < polys[0].area < 200.0  # (20 - 2*0.1) * (10 - 2*0.1) = 194.04


def test_scale_regression():
    """Verify scale regression correctly infers scale factor."""
    clean_path = os.path.join(SAMPLE_DIR, "case_0000_clean.dxf")
    ingested = ingest_dxf(clean_path)
    res = regress_drawing_scale(ingested, ingested["lines"], ingested["circles"])
    assert res["scale"] == 1.0
    assert res["scale_text"] == "1:1"


def test_benchmark_cases_ingest_and_loop_closure():
    """Run ingestion, classification, stitching, and loop building across all sample cases."""
    gt_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.gt.json")))
    assert len(gt_files) == 12, f"Expected 12 ground truth files, found {len(gt_files)}"
    
    success_loops = 0
    total_gaps_closed = 0
    
    for gtf in gt_files:
        base = gtf[:-8]
        case_name = os.path.basename(base)
        dxf_file = base + ".dxf"
        
        # 1. Ingest
        ingested = ingest_dxf(dxf_file)
        # 2. Classify
        classified = classify_and_filter_entities(ingested)
        # 3. Merge collinear fragments
        merged_lines = merge_collinear_lines(classified["lines"])
        # 4. Adaptive stitch
        stitched_lines, stats = stitch_endpoints(merged_lines)
        total_gaps_closed += stats["gaps_stitched"]
        # 5. Build loops
        polys, _ = build_closed_loops(stitched_lines, classified["arcs"])
        
        if len(polys) >= 1:
            success_loops += 1
            
    print(f"\n[Test Benchmark] Loop Closure: {success_loops}/12 cases formed closed planar profiles.")
    print(f"[Test Benchmark] Total endpoint gaps healed: {total_gaps_closed}")
    assert success_loops >= 10, f"Expected at least 10/12 cases to form closed loops, got {success_loops}"


if __name__ == "__main__":
    test_dxf_ingest_clean_structure()
    test_collinear_line_merging()
    test_adaptive_endpoint_stitching()
    test_scale_regression()
    test_benchmark_cases_ingest_and_loop_closure()
    print("\nAll DXF Ingest & Normalization tests PASSED!")
