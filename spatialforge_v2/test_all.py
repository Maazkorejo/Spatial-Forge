"""
Run:  python test_all.py
End-to-end checks with no VLM and no GPU. Every claim in the README that says "tested" is asserted here.
"""
import json
import os
import tempfile
import time

import numpy as np
import trimesh

import fit as F
from evaluate import evaluate_case
from examples import sewing_machine
from export import build_body, export_all
from features import Detection, place_features
from geometry import part_mesh
from multiview import multiview_check, render_views
from pipeline import run
from raster import part_visibility_iou, rasterize
from schema import Assembly, parse_assembly
from solver import geometry_report, resolve
from synth import perturb, synthetic_case

OK = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    OK.append(bool(cond))


truth, CAM = sewing_machine(), F.Camera(-35, 25)
case = synthetic_case(truth, CAM)
target = F.Target(case["mask"], S=160)
guess = perturb(truth, np.random.default_rng(3))


def test_schema():
    d = json.loads(truth.model_dump_json(exclude_none=True))
    d["parts"][2]["profile"] = d["parts"][2]["profile"][::-1]                       # clockwise
    a, e = parse_assembly(json.dumps(d)); check("schema rejects clockwise profile", a is None and any("counter" in x for x in e))
    d = json.loads(truth.model_dump_json(exclude_none=True)); d["parts"][0]["fillet"] = 0
    a, e = parse_assembly(json.dumps(d)); check("schema rejects rounded_box without fillet", a is None)
    d = json.loads(truth.model_dump_json(exclude_none=True)); d["features"][0]["on"]["part"] = "ghost"
    a, e = parse_assembly(json.dumps(d)); check("schema rejects feature on unknown part", a is None)
    d = json.loads(truth.model_dump_json(exclude_none=True)); d["parts"][2]["profile"] = [[0, 0], [1, 1], [1, 0], [0, 1]]
    a, e = parse_assembly(json.dumps(d)); check("schema rejects self-crossing profile", a is None)
    check("example geometry has no errors", not geometry_report(truth)["errors"])


def test_meshes():
    solids = resolve(truth)
    ok = all(trimesh.Trimesh(*part_mesh(s), process=False).is_watertight for s in solids.values())
    check("all part meshes watertight", ok)


def test_fit_and_features():
    t = time.time()
    fr = F.fit(guess, target)
    check("fit improves IoU", fr.iou > fr.init_iou + 0.1 and fr.iou > 0.93, f"{fr.init_iou:.2f}->{fr.iou:.2f} in {time.time()-t:.0f}s")
    check("camera recovered within 6 deg", abs(fr.camera.az - CAM.az) < 6 and abs(fr.camera.el - CAM.el) < 6,
          f"az {fr.camera.az:.0f} el {fr.camera.el:.0f}")
    fitted = F.apply_fit(guess, fr)
    frame = F.make_frame(F._project(resolve(fitted), fr.camera), target)
    feats, rej = place_features(fitted, fr.camera, frame, target, case["detections"])
    check("all 5 features placed, none rejected", len(feats) == 5 and not rej, str(rej))
    good = 0
    for f in feats:
        g = next(t for t in truth.features if t.kind == f.kind and t.on.part == f.on.part and t.on.face == f.on.face)
        du = abs(f.on.u - g.on.u) + abs(f.on.v - g.on.v)
        size = (f.diameter.value / g.diameter.value) if f.diameter else (f.size_u.value / g.size_u.value)
        good += (du < 0.25) and (0.6 < size < 1.6)
        print(f"     {f.id:9s} on {f.on.part}/{f.on.face} u={f.on.u:.2f} v={f.on.v:.2f}  |du|+|dv|={du:.2f}  size x{size:.2f}  verified={f.verified}")
    check("features on right part/face, position error < 0.25, size within x0.6..1.6", good == 5, f"{good}/5")
    return fitted, fr, frame


def test_part_iou(fitted, fr, frame):
    solids = resolve(truth)
    meshes = {p: part_mesh(s) for p, s in solids.items() if s.op == "union"}
    fr_true = F.make_frame(F._project(solids, CAM), target)
    ids, names, _, _ = rasterize(meshes, CAM, fr_true)
    own = {n: (ids == i).astype(np.uint8) for i, n in enumerate(names)}
    sc = part_visibility_iou(ids, names, own)
    check("per-part IoU of truth vs its own visibility masks == 1", min(sc.values()) > 0.999)
    bad_meshes = {p: part_mesh(s) for p, s in resolve(guess).items() if s.op == "union"}
    ids2, names2, _, _ = rasterize(bad_meshes, CAM, F.make_frame(F._project(resolve(guess), CAM), target))
    sc2 = part_visibility_iou(ids2, names2, own)
    check("per-part IoU drops for a wrong guess (metric discriminates)", np.mean(list(sc2.values())) < 0.95, f"mean {np.mean(list(sc2.values())):.2f}")


def test_export(fitted):
    out = tempfile.mkdtemp()
    info = export_all(truth, out, "t")
    check("STL watertight", info["watertight"], f"{info['n_triangles']} tris")
    check("no mass without anchor", info["mass_g_al6061"] is None and "relative" in info["units"])
    from schema import Anchor
    anch = truth.model_copy(update={"anchor": Anchor(part="base", dim="width_x", mm=300.0)})
    info2 = export_all(anch, out, "t_mm")
    check("anchored export is in mm with mass", info2["units"] == "mm" and info2["mass_g_al6061"] > 0,
          f"{info2['mass_g_al6061']:.0f} g Al6061")
    try:
        import cadquery as cq
        from cad_export import export_step
        s = export_step(truth, out, "t")
        back = cq.importers.importStep(s["step"])
        mesh_vol = build_body(truth).volume
        check("STEP valid and re-importable", s["valid"] and back.val().isValid())
        check("STEP volume within 2% of mesh volume", abs(s["volume"] / mesh_vol - 1) < 0.02, f"{s['volume']/mesh_vol:.3f}")
    except ImportError:
        check("STEP export test skipped (cadquery not installed)", True)


class Scripted:
    """Mock VLM: routes by prompt content."""
    def __init__(self, guess_json):
        self.guess, self.calls, self.first = guess_json, [], True

    def complete(self, system, user, images, schema=None, temperature=0.2):
        self.calls.append(user[:40])
        if "plausibly depict" in user:
            return json.dumps({"plausible": True, "score": 0.8, "issues": []})
        if self.first:
            self.first = False
            return "sorry, here is your answer: not json"
        return self.guess


def test_multiview_and_pipeline():
    png = render_views(truth)
    check("contact sheet renders", len(png) > 2000)
    v = multiview_check(Scripted(""), truth, png)
    check("multiview parses reviewer JSON", v["plausible"] is True and v["score"] == 0.8)

    vlm = Scripted(guess.model_dump_json(exclude_none=True, exclude={"features"}))
    logs = []
    res = run(case["image"], vlm, mask=case["mask"], detector=lambda img: case["detections"],
              n_candidates=1, max_repairs=0, maxiter=20, log=logs.append)
    check("pipeline status ok", res.status == "ok", f"IoU {res.iou:.2f}")
    check("pipeline recovered from a bad first VLM output (retry path)", any("attempt 0" in x for x in logs), str(logs[:2]))
    check("pipeline placed features from detector", len(res.assembly.features) == 5, f"{len(res.assembly.features)}")
    check("multi-view check recorded", res.view_check and res.view_check["plausible"] is True)
    return res


def test_evaluate(res):
    perfect = evaluate_case(truth, truth, CAM)
    check("evaluation: truth vs truth is perfect", perfect["part_f1"] == 1 and perfect["silhouette_iou"] > 0.95
          and perfect["feature_p"] == 1 and perfect["feature_r"] == 1, str({k: round(float(v), 2) for k, v in perfect.items()}))
    m = evaluate_case(res.assembly, truth, CAM)
    print("     pipeline result vs truth:", {k: (round(float(v), 3) if not isinstance(v, bool) else v) for k, v in m.items()})
    check("evaluation: pipeline result has right structure and features", m["part_f1"] == 1 and m["feature_r"] >= 0.8 and m["valid"])


if __name__ == "__main__":
    test_schema()
    test_meshes()
    fitted, fr, frame = test_fit_and_features()
    test_part_iou(fitted, fr, frame)
    test_export(fitted)
    res = test_multiview_and_pipeline()
    test_evaluate(res)
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)
