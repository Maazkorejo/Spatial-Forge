"""
Full synthetic demo, no VLM needed:  python demo.py [outdir]
  truth assembly -> fake sketch (mask + detector boxes) -> corrupted guess -> fit -> place features
  -> exports (STL, GLB, STEP, JSON) + overlay.png + shaded.png + views.png
"""
import os
import sys

import cv2
import numpy as np

import fit as F
from cad_export import export_step
from examples import sewing_machine
from export import export_all
from features import place_features
from multiview import render_views
from raster import rasterize, shade
from multiview import scene_meshes
from solver import resolve
from synth import perturb, synthetic_case


def main(out="out"):
    os.makedirs(out, exist_ok=True)
    truth, cam_true = sewing_machine(), F.Camera(-35, 25)
    case = synthetic_case(truth, cam_true)
    target = F.Target(case["mask"], S=160)
    guess = perturb(truth, np.random.default_rng(3))

    fr = F.fit(guess, target)
    print(f"silhouette IoU {fr.init_iou:.3f} -> {fr.iou:.3f}; camera az={fr.camera.az:.0f} el={fr.camera.el:.0f} "
          f"(true {cam_true.az}, {cam_true.el})")
    fitted = F.apply_fit(guess, fr)
    frame = F.make_frame(F._project(resolve(fitted), fr.camera), target)
    feats, rej = place_features(fitted, fr.camera, frame, target, case["detections"])
    final = fitted.model_copy(update={"features": feats})
    print(f"features placed: {[f.id for f in feats]}  rejected: {rej}")
    for pid, f, d in [(p.id, k, d) for p in final.parts for k, d in p.dims().items()][:8]:
        print(f"   {pid}.{f:9s} {d.value:6.1f}u  [{d.source}]")

    info = export_all(final, out, "model")
    step = export_step(final, out, "model")
    print("mesh :", {k: info[k] for k in ("stl", "glb", "watertight", "units")})
    print("step :", step)

    cv2.imwrite(os.path.join(out, "overlay.png"), cv2.imdecode(np.frombuffer(F.make_overlay(fr.rendered, target), np.uint8), 1))
    meshes, solids, n_parts = scene_meshes(final)
    ids, names, nb, _ = rasterize(meshes, fr.camera, F.make_frame(F._project(solids, fr.camera), F.Target(np.ones((100, 160), np.uint8), S=320)))
    tint = {k: (40, 130, 235) for k, n in enumerate(names) if k >= n_parts}
    cv2.imwrite(os.path.join(out, "shaded.png"), shade(ids, nb, fr.camera, tint))
    with open(os.path.join(out, "views.png"), "wb") as fh:
        fh.write(render_views(final))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
