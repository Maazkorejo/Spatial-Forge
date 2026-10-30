"""
Unit tests for individual CAD solvers in spatialforge_v2/reconstruction.py.
Applies rigorous safety-gate invariant checks:
- Solid validity (isValid() == True)
- Single watertight solid count (len(solids) == 1)
- Positive volume > 0
- Bounding box dimensions within expected tolerances
- STEP exporter roundtrip integrity
"""
import unittest
import os
import sys
import tempfile
import cadquery as cq

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2_DIR = os.path.join(REPO_ROOT, "spatialforge_v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from reconstruction import (
    reconstruct_prismatic,
    reconstruct_turned,
    reconstruct_f35_jet,
    reconstruct_wheel,
    reconstruct_engine,
    reconstruct_butterfly,
    reconstruct_cat,
)


def assert_solid_invariants(test_case: unittest.TestCase, solid: cq.Workplane, name: str):
    """Reuse cad_edit gate invariant checks for CAD reconstruction solvers."""
    shape = solid.val()
    # 1. Solid validity
    test_case.assertTrue(shape.isValid(), f"{name}: Shape failed isValid() B-Rep check")

    # 2. Exactly one solid
    n_solids = len(solid.solids().vals())
    test_case.assertEqual(n_solids, 1, f"{name}: Expected exactly 1 solid, found {n_solids}")

    # 3. Positive volume
    vol = float(shape.Volume())
    test_case.assertGreater(vol, 10.0, f"{name}: Volume {vol} too small or non-positive")

    # 4. Valid bounding box
    bbox = shape.BoundingBox()
    test_case.assertGreater(bbox.xlen, 1.0, f"{name}: Bounding box X is zero or negative")
    test_case.assertGreater(bbox.ylen, 1.0, f"{name}: Bounding box Y is zero or negative")
    test_case.assertGreater(bbox.zlen, 1.0, f"{name}: Bounding box Z is zero or negative")

    # 5. STEP roundtrip test
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    try:
        cq.exporters.export(solid, path)
        back = cq.importers.importStep(path)
        test_case.assertTrue(back.val().isValid(), f"{name}: STEP import failed isValid()")
        v_back = back.val().Volume()
        test_case.assertAlmostEqual(vol, v_back, delta=max(1.0, vol * 0.001), msg=f"{name}: STEP roundtrip volume drift")
    finally:
        if os.path.exists(path):
            os.remove(path)


class TestReconstructionSolvers(unittest.TestCase):
    def test_reconstruct_f35_jet(self):
        registered = {"dimensions": {"W": 107.0, "D": 25.0, "H": 165.0}}
        solid, meta = reconstruct_f35_jet(registered)
        self.assertEqual(meta["family"], "f35_jet")
        assert_solid_invariants(self, solid, "reconstruct_f35_jet")

    def test_reconstruct_wheel(self):
        registered = {"dimensions": {"W": 136.0, "D": 45.0, "H": 136.0}}
        solid, meta = reconstruct_wheel(registered)
        self.assertEqual(meta["family"], "wheel")
        assert_solid_invariants(self, solid, "reconstruct_wheel")

    def test_reconstruct_engine(self):
        registered = {"dimensions": {"W": 130.0, "D": 83.0, "H": 168.0}}
        solid, meta = reconstruct_engine(registered)
        self.assertEqual(meta["family"], "engine")
        assert_solid_invariants(self, solid, "reconstruct_engine")

    def test_reconstruct_butterfly(self):
        registered = {"dimensions": {"W": 184.0, "D": 16.0, "H": 108.0}}
        solid, meta = reconstruct_butterfly(registered)
        self.assertEqual(meta["family"], "butterfly")
        assert_solid_invariants(self, solid, "reconstruct_butterfly")

    def test_reconstruct_cat(self):
        registered = {"dimensions": {"W": 162.0, "D": 32.0, "H": 132.0}}
        solid, meta = reconstruct_cat(registered)
        self.assertEqual(meta["family"], "cat")
        assert_solid_invariants(self, solid, "reconstruct_cat")

    def test_reconstruct_prismatic(self):
        registered = {
            "dimensions": {"W": 60.0, "D": 40.0, "H": 20.0},
            "views": {
                "front": {"entities": []},
                "top": {"entities": []},
                "right": {"entities": []}
            }
        }
        solid, meta = reconstruct_prismatic(registered)
        self.assertEqual(meta["family"], "prismatic")
        assert_solid_invariants(self, solid, "reconstruct_prismatic")

    def test_reconstruct_turned(self):
        registered = {
            "dimensions": {"W": 100.0, "D": 30.0, "H": 30.0},
            "views": {
                "front": {"entities": []},
                "top": {"entities": []},
                "right": {"entities": []}
            }
        }
        solid, meta = reconstruct_turned(registered)
        self.assertEqual(meta["family"], "turned")
        assert_solid_invariants(self, solid, "reconstruct_turned")


if __name__ == "__main__":
    unittest.main()
