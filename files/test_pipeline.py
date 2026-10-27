"""Pipeline smoke test with a mock VLM (first output is invalid JSON, then a valid but imperfect guess)."""
import numpy as np
import fit as F
from demo import perturb
from examples import sewing_machine
from pipeline import run
from solver import resolve
from vlm import MockVLM

truth, cam = sewing_machine(), F.Camera(-35, 25)
polys = F._project(resolve(truth), cam)
allp = np.vstack(polys); w, h = allp.max(0) - allp.min(0)
t0 = F.Target(np.ones((50, 50), np.uint8)); t0.W, t0.H = 134 * w / max(w, h), 134 * h / max(w, h)
mask = F.render(truth, cam, t0) * 255
image = np.dstack([255 - mask] * 3).astype(np.uint8)

guess = perturb(truth, np.random.default_rng(1))
vlm = MockVLM(["not json at all", guess.model_dump_json(exclude_none=True)])
res = run(image, vlm, mask=mask, n_candidates=1, max_repairs=1)
print("status:", res.status, " IoU: %.3f" % res.iou, " parts:", len(res.assembly.parts), " overlay bytes:", len(res.overlay_png))
