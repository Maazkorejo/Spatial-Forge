"""
Unit & Golden Regression Tests for Containment Tree CAD Profile Engine.
Validates:
1. Swashplate DIF-049 composite ring + 3 tabs (reprojection F1 and recall >= 95%).
2. Pure disc / washer with concentric hole.
3. Plate with bolt circle of 6 through-holes.
4. Two disconnected parts in one drawing (returns 2 distinct solids).
5. Nested island-in-hole depth parity (even-odd CAD rule).
"""
import math
import os
import sys
from shapely.geometry import Polygon, Point

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

DRAWING_GEN_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "drawing_gen", "drawing_gen"))
if DRAWING_GEN_DIR not in sys.path:
    sys.path.insert(0, DRAWING_GEN_DIR)

from reconstruction import build_containment_profile
from dxf_pipeline import process_dxf_bytes


def test_pure_disc_washer():
    """Concentric disc and bore: Depth 0 solid, Depth 1 hole."""
    disc_outer = Point(50, 50).buffer(30, resolution=32)
    disc_hole = Point(50, 50).buffer(10, resolution=32)
    solids = build_containment_profile([disc_outer, disc_hole])
    assert len(solids) == 1, f"Expected 1 solid, got {len(solids)}"
    assert len(solids[0].interiors) == 1, f"Expected 1 hole, got {len(solids[0].interiors)}"
    expected_area = math.pi * (30**2 - 10**2)
    assert abs(solids[0].area - expected_area) / expected_area < 0.02


def test_plate_with_bolt_circle():
    """Plate containing a 6-hole bolt circle: Depth 0 plate, Depth 1 holes."""
    plate = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    bolt_holes = [
        Point(50 + 30 * math.cos(i * math.pi / 3), 50 + 30 * math.sin(i * math.pi / 3)).buffer(4, resolution=16)
        for i in range(6)
    ]
    solids = build_containment_profile([plate] + bolt_holes)
    assert len(solids) == 1, f"Expected 1 solid, got {len(solids)}"
    assert len(solids[0].interiors) == 6, f"Expected 6 holes, got {len(solids[0].interiors)}"


def test_two_disconnected_parts():
    """Two genuinely disconnected parts in one drawing: returns 2 distinct solids."""
    part1 = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
    part2 = Polygon([(60, 0), (90, 0), (90, 30), (60, 30)])
    solids = build_containment_profile([part1, part2])
    assert len(solids) == 2, f"Expected 2 disconnected solids, got {len(solids)}"
    assert abs(solids[0].area - 900.0) < 1.0
    assert abs(solids[1].area - 900.0) < 1.0


def test_nested_island_parity():
    """
    Even-odd CAD fill rule:
    Depth 0: Outer ring solid
    Depth 1: Annular hole
    Depth 2: Floating inner island solid
    Depth 3: Hole inside floating island
    """
    c0 = Point(0, 0).buffer(50, resolution=32)  # Depth 0: solid
    c1 = Point(0, 0).buffer(40, resolution=32)  # Depth 1: hole
    c2 = Point(0, 0).buffer(25, resolution=32)  # Depth 2: solid island
    c3 = Point(0, 0).buffer(10, resolution=32)  # Depth 3: hole inside island

    solids = build_containment_profile([c0, c1, c2, c3])
    # Expect 2 solid bodies: outer annular ring and inner floating ring
    assert len(solids) == 2, f"Expected 2 solid bodies from parity nesting, got {len(solids)}"


def test_swashplate_composite_dxf():
    """
    Real-world test on DIF-049 Helicopter Swashplate:
    Hub circle + 3 horn arms unioned into a single body with 4 through-holes.
    Reprojection verification must achieve status == 'ok' with recall >= 95% and F1 >= 95%.
    """
    repo_root = os.path.abspath(os.path.join(CUR_DIR, ".."))
    dxf_path = os.path.join(repo_root, "test_cad_models", "03_difficult_cads", "49_swashplate_control_ring.dxf")
    if not os.path.exists(dxf_path):
        print(f"Skipping: Test CAD model not found at {dxf_path}")
        return

    with open(dxf_path, "rb") as f:
        contents = f.read()

    res = process_dxf_bytes(contents, "49_swashplate_control_ring.dxf")
    assert res.get("status") == "ok", f"Expected status 'ok', got {res.get('status')}: {res.get('failure_reason')}"

    reproj = res.get("reprojection", {})
    top_view = reproj.get("views", {}).get("top", {}).get("visible", {})
    recall = top_view.get("recall", 0.0)
    f1 = top_view.get("f1", 0.0)

    print(f"\nSwashplate DIF-049 Verification: Recall={recall * 100:.1f}%, F1={f1 * 100:.1f}%")
    assert recall >= 0.95, f"Expected Recall >= 95%, got {recall * 100:.1f}%"
    assert f1 >= 0.95, f"Expected F1 >= 95%, got {f1 * 100:.1f}%"
    assert res.get("thickness_info", {}).get("value_mm") == 3.0, "Expected 3.0 mm thickness"


if __name__ == "__main__":
    test_pure_disc_washer()
    print("test_pure_disc_washer PASSED")
    test_plate_with_bolt_circle()
    print("test_plate_with_bolt_circle PASSED")
    test_two_disconnected_parts()
    print("test_two_disconnected_parts PASSED")
    test_nested_island_parity()
    print("test_nested_island_parity PASSED")
    test_swashplate_composite_dxf()
    print("test_swashplate_composite_dxf PASSED")
    print("\nALL CONTAINMENT TREE TESTS PASSED SUCCESSFULLY!")
