"""
Self-test with no VLM:
  1. render a known assembly (ground truth) from a known camera -> fake "sketch mask"
  2. simulate a VLM guess: perturb every dimension by up to ~-25%/+30%, re-centre the ranges
  3. run the fit and report IoU gain, dimension recovery and sensitivity labels
Run:  python demo.py
"""
import time

import numpy as np

import fit as F
from examples import sewing_machine
from schema import Assembly
from solver import geometry_report, resolve


def perturb(truth: Assembly, rng, lo=0.75, hi=1.3, tol=0.4) -> Assembly:
    a = truth.model_copy(deep=True)
    for p in a.parts:
        for _, d in p.dims().items():
            v = d.value * rng.uniform(lo, hi)
            d.value, d.min, d.max = v, v * (1 - tol), v * (1 + tol)
            d.source = "inferred"
    return a


def main(seed=3):
    rng = np.random.default_rng(seed)
    truth = sewing_machine()
    cam_true = F.Camera(-35, 25)

    # fake sketch: render the truth in a canvas with the same extents
    polys = F._project(resolve(truth), cam_true)
    allp = np.vstack(polys)
    w, h = allp.max(0) - allp.min(0)
    blank = np.zeros((int(1000 * h / max(w, h)) + 2, int(1000 * w / max(w, h)) + 2), np.uint8) + 1
    t0 = F.Target(blank, S=160)
    t0.W, t0.H = 160 * 0.84 * w / max(w, h), 160 * 0.84 * h / max(w, h)
    sketch = F.render(truth, cam_true, t0)
    target = F.Target(sketch, S=160)

    guess = perturb(truth, rng)
    print("geometry report:", geometry_report(guess)["errors"] or "ok")

    t = time.time()
    fr = F.fit(guess, target)
    print(f"fit: {time.time() - t:.1f}s, {fr.n_evals} renders")
    print(f"IoU {fr.init_iou:.3f} -> {fr.iou:.3f}   camera az={fr.camera.az:.0f} el={fr.camera.el:.0f} "
          f"(true az={cam_true.az} el={cam_true.el})")

    out = F.apply_fit(guess, fr)
    # compare ratios (global scale is unobservable), normalised by base width
    def norm(asm):
        return {(p.id, f): d.value / asm.parts[0].width_x.value for p in asm.parts for f, d in p.dims().items()}
    nt, ng, nf = norm(truth), norm(guess), norm(out)
    print(f"\n{'part.dim':22s} {'truth':>7s} {'guess':>7s} {'fitted':>7s} {'err_before':>10s} {'err_after':>9s}  label")
    tot_b = tot_a = 0
    for p in out.parts:
        for f, d in p.dims().items():
            k = (p.id, f)
            eb, ea = abs(ng[k] / nt[k] - 1), abs(nf[k] / nt[k] - 1)
            tot_b += eb; tot_a += ea
            print(f"{p.id + '.' + f:22s} {nt[k]:7.3f} {ng[k]:7.3f} {nf[k]:7.3f} {eb:10.0%} {ea:9.0%}  {d.source}")
    k = len(nt)
    print(f"\nmean relative error: {tot_b / k:.0%} -> {tot_a / k:.0%}")


if __name__ == "__main__":
    main()
