"""
API Contract Freeze Tests for SpatialForge.
Verifies that all API endpoints adhere strictly to the frozen request and response schemas.
"""
import unittest
import os
import sys
import json
import urllib.request
import io

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA_PATH = os.path.join(REPO_ROOT, "tests", "fixtures", "api_contract_schema.json")
BASE_URL = "http://127.0.0.1:8085"


def _post_json(path: str, data: dict) -> dict:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _get_json(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def _post_multipart(path: str, filename: str, file_bytes: bytes, field_name: str = "file") -> dict:
    boundary = "----WebKitFormBoundarySpatialForgeContractTest"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode())
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


class TestApiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            cls.schema = json.load(f)["endpoints"]

    def test_cad_edit_init_contract(self):
        ep_schema = self.schema["/api/v1/cad_edit/init"]
        resp = _post_json("/api/v1/cad_edit/init", {"session_id": "contract_sess", "W": 40, "D": 20, "H": 10})
        for req_field in ep_schema["required_response_fields"]:
            self.assertIn(req_field, resp, f"Field '{req_field}' missing from /api/v1/cad_edit/init response")

    def test_cad_edit_state_contract(self):
        ep_schema = self.schema["/api/v1/cad_edit/state"]
        resp = _get_json("/api/v1/cad_edit/state?session_id=contract_sess")
        for req_field in ep_schema["required_response_fields"]:
            self.assertIn(req_field, resp, f"Field '{req_field}' missing from /api/v1/cad_edit/state response")

    def test_cad_edit_propose_contract(self):
        ep_schema = self.schema["/api/v1/cad_edit/propose"]
        hole_op = [{"op": "add", "feature": {"kind": "hole", "face": "top", "d": 4.0}}]
        resp = _post_json("/api/v1/cad_edit/propose", {"session_id": "contract_sess", "ops": hole_op})
        for req_field in ep_schema["required_response_fields"]:
            self.assertIn(req_field, resp, f"Field '{req_field}' missing from /api/v1/cad_edit/propose response")

    def test_cad_edit_apply_contract(self):
        ep_schema = self.schema["/api/v1/cad_edit/apply"]
        hole_op = [{"op": "add", "feature": {"kind": "hole", "face": "top", "d": 4.0}}]
        resp = _post_json("/api/v1/cad_edit/apply", {"session_id": "contract_sess", "ops": hole_op})
        for req_field in ep_schema["required_response_fields"]:
            self.assertIn(req_field, resp, f"Field '{req_field}' missing from /api/v1/cad_edit/apply response")

    def test_cad_edit_undo_redo_contract(self):
        undo_schema = self.schema["/api/v1/cad_edit/undo"]
        resp_undo = _post_json("/api/v1/cad_edit/undo", {"session_id": "contract_sess"})
        for req_field in undo_schema["required_response_fields"]:
            self.assertIn(req_field, resp_undo, f"Field '{req_field}' missing from /api/v1/cad_edit/undo response")

        redo_schema = self.schema["/api/v1/cad_edit/redo"]
        resp_redo = _post_json("/api/v1/cad_edit/redo", {"session_id": "contract_sess"})
        for req_field in redo_schema["required_response_fields"]:
            self.assertIn(req_field, resp_redo, f"Field '{req_field}' missing from /api/v1/cad_edit/redo response")

    def test_image_synthesis_contract(self):
        ep_schema = self.schema["/api/v1/image"]
        sketch_path = os.path.join(REPO_ROOT, "tests", "fixtures", "sample_sketch.png")
        with open(sketch_path, "rb") as f:
            img_bytes = f.read()
        resp = _post_multipart("/api/v1/image", "sample_sketch.png", img_bytes)
        for req_field in ep_schema["required_response_fields"]:
            self.assertIn(req_field, resp, f"Field '{req_field}' missing from /api/v1/image response")
        mesh_body = resp["meshes"]["body"]
        for mesh_f in ep_schema["mesh_body_fields"]:
            self.assertIn(mesh_f, mesh_body, f"Mesh field '{mesh_f}' missing from image response body")

    def test_dxf_synthesize_contract(self):
        ep_schema = self.schema["/api/v1/synthesize"]
        dxf_sample = os.path.join(REPO_ROOT, "drawing_gen", "drawing_gen", "sample_cases", "case_0000_clean.dxf")
        with open(dxf_sample, "rb") as f:
            dxf_bytes = f.read()
        resp = _post_multipart("/api/v1/synthesize", "case_0000_clean.dxf", dxf_bytes)
        for req_field in ep_schema["required_response_fields"]:
            self.assertIn(req_field, resp, f"Field '{req_field}' missing from /api/v1/synthesize response")
        mesh_body = resp["meshes"]["body"]
        for mesh_f in ep_schema["mesh_body_fields"]:
            self.assertIn(mesh_f, mesh_body, f"Mesh field '{mesh_f}' missing from DXF response body")


if __name__ == "__main__":
    unittest.main()
