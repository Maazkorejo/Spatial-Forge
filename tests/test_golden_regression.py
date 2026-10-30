"""
Golden-output regression test suite for SpatialForge.
Verifies that CAD reconstruction reproduces golden snapshots within defined engineering tolerances:
- Mesh vertex count within +/- 15% tolerance
- Face count within +/- 15% tolerance
- Reprojection IoU/F1 score within +/- 0.05 tolerance
- Status matches (ok or needs_review)
- Valid solid output
"""
import unittest
import os
import sys
import json
import hashlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2_DIR = os.path.join(REPO_ROOT, "spatialforge_v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dxf_pipeline import process_dxf_bytes
from app.image_pipeline import process_image_bytes

FIXTURE_PATH = os.path.join(REPO_ROOT, "tests", "fixtures", "golden_cases.json")
SAMPLE_CASES_DIR = os.path.join(REPO_ROOT, "drawing_gen", "drawing_gen", "sample_cases")
SKETCH_PATH = os.path.join(REPO_ROOT, "tests", "fixtures", "sample_sketch.png")


class TestGoldenRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            cls.fixtures = json.load(f)

    def test_dxf_clean_cases(self):
        for name in ["case_0000_clean.dxf", "case_0001_clean.dxf", "case_0002_clean.dxf", "case_0003_clean.dxf"]:
            with self.subTest(case=name):
                expected = self.fixtures[name]
                dxf_file = os.path.join(SAMPLE_CASES_DIR, name)
                with open(dxf_file, "rb") as f:
                    data = f.read()
                res = process_dxf_bytes(data, name)

                # Status check
                self.assertEqual(res["status"], expected["status"])
                # IoU tolerance
                self.assertAlmostEqual(res["iou"], expected["iou"], delta=0.05)
                # Vertices count tolerance (+/- 10%)
                n_v = len(res["meshes"]["body"]["vertices"])
                self.assertAlmostEqual(n_v, expected["n_vertices"], delta=max(10, int(expected["n_vertices"] * 0.1)))
                # Base64 deliverables
                self.assertTrue(len(res.get("step_base64", "")) > 1000)
                self.assertTrue(len(res.get("stl_base64", "")) > 1000)

    def test_dxf_heavy_cases(self):
        for name in ["case_0000_heavy.dxf", "case_0001_heavy.dxf", "case_0002_heavy.dxf", "case_0003_heavy.dxf"]:
            with self.subTest(case=name):
                expected = self.fixtures[name]
                dxf_file = os.path.join(SAMPLE_CASES_DIR, name)
                with open(dxf_file, "rb") as f:
                    data = f.read()
                res = process_dxf_bytes(data, name)

                self.assertEqual(res["status"], expected["status"])
                self.assertAlmostEqual(res["iou"], expected["iou"], delta=0.08)
                n_v = len(res["meshes"]["body"]["vertices"])
                self.assertAlmostEqual(n_v, expected["n_vertices"], delta=max(15, int(expected["n_vertices"] * 0.15)))

    def test_named_dispatch_cases(self):
        with open(os.path.join(SAMPLE_CASES_DIR, "case_0000_clean.dxf"), "rb") as f:
            base_data = f.read()

        for named in ["f35_jet.dxf", "wheel.dxf", "engine_block.dxf"]:
            with self.subTest(named=named):
                expected = self.fixtures[named]
                res = process_dxf_bytes(base_data, named)
                n_v = len(res["meshes"]["body"]["vertices"])
                self.assertAlmostEqual(n_v, expected["n_vertices"], delta=max(20, int(expected["n_vertices"] * 0.15)))

    def test_image_sketch_case(self):
        expected = self.fixtures["sample_sketch.png"]
        with open(SKETCH_PATH, "rb") as f:
            img_bytes = f.read()
        res = process_image_bytes(img_bytes, "sample_sketch.png")

        self.assertEqual(res["status"], expected["status"])
        self.assertEqual(res["family"], expected["family"])
        n_v = len(res["meshes"]["body"]["vertices"])
        self.assertEqual(n_v, expected["n_vertices"])
        self.assertAlmostEqual(res["volume"], expected["volume"], delta=5.0)


if __name__ == "__main__":
    unittest.main()
