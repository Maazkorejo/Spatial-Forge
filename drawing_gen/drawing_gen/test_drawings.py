"""
Run:  python test_drawings.py        (~1 min, CPU only)
Verifies the benchmark generator itself. If the generator is wrong, every later measurement is wrong.
"""
import json
import math
import os
import tempfile

import cadquery as cq
import numpy as np

from draw import SCALES, build_drawing
from dxf_io import read_dxf, write_dxf
from generate import make_case, save_case
from metrics import edge_prf, flag_stats, hull_iou, volume_iou
from noise import apply_noise
from parts import make_part
from project import bbox, project_view

OK = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    OK.append(bool(cond))


def test_projection_conventions():
    blk = cq.Workplane("XY").box(40, 20, 10, centered=False).cut(cq.Workplane("XY").box(10, 6, 4, centered=False).translate((5, 0, 3)))
    f, t, r = (project_view(blk.val(), v) for v in ("front", "top", "right"))
    bb = lambda v: [round(x, 3) for x in bbox(v["visible"])]
    check("front view = (x, z)", bb(f) == [0, 0, 40, 10], str(bb(f)))
    check("top view = (x, y), y up", bb(t) == [0, 0, 40, 20], str(bb(t)))
    check("right view = (y, z)", bb(r) == [0, 0, 20, 10], str(bb(r)))
    check("pocket cut into the FRONT face is drawn visible in the front view", len(f["visible"]) == 8, str(len(f["visible"])))
    check("same pocket is hidden-only in the top view", len(t["hidden"]) == 3 and len(t["visible"]) == 4,
          f"hidden {len(t['hidden'])} visible {len(t['visible'])}")
    check("coincident hidden lines are suppressed (visible wins)", all(
        not (h["t"] == "line" and abs(h["p"][1]) < 1e-6 and abs(h["p"][3]) < 1e-6) for h in t["hidden"]))
    edge_on = cq.Workplane("XY").circle(3).extrude(10)
    fr = project_view(edge_on.val(), "front")
    check("circle seen edge-on becomes lines, not 24 fragments", sum(p["t"] == "poly" for p in fr["visible"] + fr["hidden"]) == 0)
    tp = project_view(edge_on.val(), "top")
    c = [p for p in tp["visible"] if p["t"] == "circle"]
    check("cylinder along z is a true circle in the top view", len(c) >= 1 and abs(c[0]["r"] - 3) < 1e-6)


def test_parts():
    bad, fams = 0, set()
    for s in range(40):
        b, sp = make_part(np.random.default_rng(s))
        bad += not (sp["valid"] and sp["volume"] > 100 and len(b.solids().vals()) == 1)
        fams.add(sp["family"])
    check("40 random parts are valid single solids", bad == 0 and fams == {"prismatic", "turned"}, f"bad={bad}")


def _ranges(d, view, roles=("visible",)):
    xs, ys = [], []
    for e in d["ents"]:
        if e.get("view") == view and e["role"] in roles and e["kind"] == "line":
            xs += [e["g"]["p"][0], e["g"]["p"][2]]
            ys += [e["g"]["p"][1], e["g"]["p"][3]]
    return min(xs), max(xs), min(ys), max(ys)


def test_layout():
    ok3 = ok1 = okscale = oksc = True
    for seed in range(8):
        b, sp = make_part(np.random.default_rng(seed), "prismatic")
        for std in ("third", "first"):
            for sc, s in SCALES.items():
                d = build_drawing(b, sp, std, sc)
                F, T, R = _ranges(d, "front"), _ranges(d, "top"), _ranges(d, "right")
                aligned = abs(F[0] - T[0]) < 1e-6 and abs(F[1] - T[1]) < 1e-6 and abs(F[2] - R[2]) < 1e-6 and abs(F[3] - R[3]) < 1e-6
                if std == "third":
                    ok3 &= aligned and T[2] > F[3] and R[0] > F[1]
                else:
                    ok1 &= aligned and T[3] < F[2] and R[1] < F[0]
                okscale &= abs((F[1] - F[0]) - sp["W"] * s) < 1e-6 and abs((F[3] - F[2]) - sp["H"] * s) < 1e-6
                oksc &= all(abs(m["drawn"] / m["value_real"] - s) < 1e-9 for m in d["meta"]["dims"])
    check("third angle: top above front, right view right of front, views aligned", ok3)
    check("first angle: top below front, right view LEFT of front, views aligned", ok1)
    check("drawn size = real size x scale (1:1, 1:2, 2:1)", okscale)
    check("dimension truth: drawn / real value = scale", oksc)


def test_dxf_roundtrip():
    b, sp = make_part(np.random.default_rng(5), "prismatic")
    for sc, s in (("1:1", 1.0), ("1:2", 0.5)):
        d = build_drawing(b, sp, "third", sc)
        p = write_dxf(d, os.path.join(tempfile.mkdtemp(), "t.dxf"))
        r = read_dxf(p)
        cnt = lambda k: sum(e["kind"] == k for e in d["ents"])
        check(f"[{sc}] DXF entity counts match the IR", len(r["lines"]) == cnt("line") and len(r["circles"]) == cnt("circle")
              and len(r["arcs"]) == cnt("arc") and len(r["texts"]) == cnt("text") and len(r["dims"]) == cnt("dim_linear") + cnt("dim_diam"),
              f"lines {len(r['lines'])}/{cnt('line')}")
        meas = sorted(round(x["measurement"], 2) for x in r["dims"])
        drawn = sorted(round(m["drawn"], 2) for m in d["meta"]["dims"])
        check(f"[{sc}] native dimension measurement == drawn length", meas == drawn)
        if s != 1.0:
            texts = sorted(x["text"] for x in r["dims"])
            check(f"[{sc}] text override carries the REAL value (measurement != text)", all(t != "<>" for t in texts) and "95" in texts, str(texts))
    check("units header is millimetres", r["insunits"] == 4)


def _endpoint_match_rate(lines):
    pts = [tuple(np.round(l["p"][i:i + 2], 4)) for l in lines for i in (0, 2)]
    from collections import Counter
    c = Counter(pts)
    return sum(1 for p in pts if c[p] > 1) / len(pts)


def test_noise():
    b, sp = make_part(np.random.default_rng(5), "prismatic")
    d = build_drawing(b, sp, "third", "1:2")
    n1, l1 = apply_noise(d, np.random.default_rng(1), "heavy")
    n2, l2 = apply_noise(d, np.random.default_rng(1), "heavy")
    check("noise is deterministic for a seed", l1 == l2 and json.dumps(n1["ents"], sort_keys=True) == json.dumps(n2["ents"], sort_keys=True))
    p = write_dxf(n1, os.path.join(tempfile.mkdtemp(), "h.dxf"))
    r, rc = read_dxf(p), read_dxf(write_dxf(d, os.path.join(tempfile.mkdtemp(), "c.dxf")))
    check("heavy: no native DIMENSION left (exploded to lines + text)", len(r["dims"]) == 0 and len(r["texts"]) > len(rc["texts"]))
    check("heavy: every entity on layer 0", {e["layer"] for e in r["lines"] + r["circles"] + r["arcs"]} == {"0"})
    check("heavy: dashed linetype gone, replaced by many short segments",
          all(e["ltype"] != "DASHED" for e in r["lines"]) and len(r["lines"]) > 1.5 * len(rc["lines"]), f"{len(rc['lines'])} -> {len(r['lines'])}")
    check("heavy: scale text removed (this seed)", l1["scale_text_dropped"] and not any("SCALE" in t["text"] for t in r["texts"]))
    ce = [{"p": e["g"]["p"]} for e in d["ents"] if e["kind"] == "line" and e["role"] == "visible"]
    ne = [{"p": e["g"]["p"]} for e in n1["ents"] if e["kind"] == "line" and e["role"] == "visible"]
    mc, mn = _endpoint_match_rate(ce), _endpoint_match_rate(ne)
    check("gaps break endpoint coincidence (baseline <1: T-junctions have no partner endpoint)", mc > 0.8 and mn < 0.4, f"{mc:.2f} -> {mn:.2f}")
    seen = None
    for sd in range(20):
        nn, ll = apply_noise(d, np.random.default_rng(sd), "heavy")
        if not ll["units_header"]:
            seen = nn
            break
    rr = read_dxf(write_dxf(seen, os.path.join(tempfile.mkdtemp(), "u.dxf"), units_header=False)) if seen else None
    check("missing-units case writes $INSUNITS = 0", rr is not None and rr["insunits"] == 0)


def test_metrics():
    c0 = make_case(7, "clean", "prismatic", "third", "1:1")
    ch, hv = make_case(7, "heavy", "prismatic", "third", "1:1"), make_case(7, "light", "prismatic", "third", "1:1")
    gt = c0["clean"]["ents"]
    check("edge F1: clean vs clean = 1", edge_prf(gt, gt)["f1"] == 1.0)
    fl = edge_prf(hv["noisy"]["ents"], gt)
    fh = edge_prf(ch["noisy"]["ents"], gt)
    check("edge F1 stays high under light and heavy noise (fragmentation-proof)", fl["f1"] > 0.97 and fh["f1"] > 0.93,
          f"light {fl['f1']:.3f}  heavy {fh['f1']:.3f}")
    hid = edge_prf(ch["noisy"]["ents"], gt, roles=("hidden",), tol=1.0)
    check("hidden-line F1 survives dash explosion (tol >= half dash gap)", hid["f1"] > 0.9, f"{hid['f1']:.3f}")
    other = make_case(11, "clean", "prismatic", "third", "1:1")["clean"]["ents"]
    fo = edge_prf(other, gt)["f1"]
    check("edge F1 discriminates a different part", fo < 0.8, f"{fo:.3f}")
    box = cq.Workplane("XY").box(40, 30, 20, centered=False)
    pk = box.cut(cq.Workplane("XY").box(15, 10, 10, centered=False).translate((10, 10, 10)))
    check("volume IoU exact on a known case", abs(volume_iou(box, pk) - pk.val().Volume() / box.val().Volume()) < 1e-6)
    fs = flag_stats([(0.99, False), (0.95, False), (0.5, True), (0.6, False), (0.8, True), (0.97, True)])
    check("flag_stats separates recall from false alarms", abs(fs["flag_recall"] - 2 / 3) < 1e-9 and abs(fs["false_alarm"] - 1 / 3) < 1e-9 and fs["silent_failures"] == 1)


def test_visual_hull_claim():
    groups = {"exact": [], "blind": [], "turned": []}
    for s in range(40):
        b, sp = make_part(np.random.default_rng(s))
        key = "exact" if sp["hull_exact"] else ("turned" if sp["family"] == "turned" else "blind")
        groups[key].append(hull_iou(b.val())["iou"])
    for k, v in groups.items():
        if v:
            print(f"     {k:7s} n={len(v):2d}  mean hull IoU={np.mean(v):.3f}  min={np.min(v):.3f}")
    check("through-features-only prismatic parts: visual hull is exact (IoU > 0.99)", np.mean(groups["exact"]) > 0.99 and np.min(groups["exact"]) > 0.97,
          f"{np.mean(groups['exact']):.3f}")
    check("blind-feature prismatic parts: visual hull over-fills (mean IoU < 0.99)", np.mean(groups["blind"]) < 0.99, f"{np.mean(groups['blind']):.3f}")
    check("turned parts: visual hull is clearly wrong (mean IoU < 0.95, rounded-square section)", np.mean(groups["turned"]) < 0.95,
          f"{np.mean(groups['turned']):.3f}")
    n_exact = len(groups["exact"])
    print(f"     => only {n_exact}/40 random parts are fully determined by their three silhouettes")


def test_generate():
    out = tempfile.mkdtemp()
    for s in range(3):
        for lv in ("clean", "light", "heavy"):
            paths = save_case(make_case(s, lv), out)
    files = os.listdir(out)
    check("generator writes dxf + step + gt.json for every case", len([f for f in files if f.endswith(".dxf")]) == 9 and
          len([f for f in files if f.endswith(".step")]) == 9 and len([f for f in files if f.endswith(".gt.json")]) == 9)
    gt = json.load(open(os.path.join(out, "case_0001_heavy.gt.json")))
    shape = cq.importers.importStep(os.path.join(out, "case_0001_heavy.step"))
    check("STEP ground truth re-imports with the recorded volume", abs(shape.val().Volume() / gt["spec"]["volume"] - 1) < 1e-4)
    check("gt.json keeps CLEAN entities with roles for scoring", {e["role"] for e in gt["clean_entities"]} >= {"visible", "hidden", "dim", "title"})


if __name__ == "__main__":
    test_projection_conventions()
    test_parts()
    test_layout()
    test_dxf_roundtrip()
    test_noise()
    test_metrics()
    test_visual_hull_claim()
    test_generate()
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)
