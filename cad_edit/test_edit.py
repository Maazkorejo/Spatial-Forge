"""
Run:  python test_edit.py     (~1 min, CPU only)
Every claim in README.md marked "tested" is asserted here.
"""
import math
import os
import tempfile

import cadquery as cq
import numpy as np

from build import build
from convert import from_spec
from gen_parts import make_part
from model import (AddOp, At, Block, Boss, BoltPattern, Chamfer, DeleteOp, FeatureTree, Fillet, Hole, PocketRect, Ref,
                   ResizeOp, SetOp, Shell, Slot, parse_ops)
from session import Session

OK = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    OK.append(bool(cond))


def T(*feats, W=40, D=20, H=10):
    return FeatureTree(base=Block(W=W, D=D, H=H), features=list(feats))


def removed(tree):
    return tree.base.W * tree.base.D * tree.base.H - build(tree).val().Volume()


def test_parse():
    ok, err = parse_ops('```json\n{"ops":[{"op":"add","feature":{"kind":"hole","face":"top","d":6}}]}\n```')
    check("parse_ops accepts fenced JSON and validates", ok and not err)
    cases = {
        "negative diameter": '{"ops":[{"op":"add","feature":{"kind":"hole","face":"top","d":-2}}]}',
        "unknown operation": '{"ops":[{"op":"explode"}]}',
        "counterbore not larger than hole": '{"ops":[{"op":"add","feature":{"kind":"hole","face":"top","d":6,"cbore_d":5,"cbore_depth":2}}]}',
        "circular boss with rect sizes": '{"ops":[{"op":"add","feature":{"kind":"boss","face":"top","shape":"circ","d":5,"size_u":3,"height":2}}]}',
        "slot shorter than wide": '{"ops":[{"op":"add","feature":{"kind":"slot","face":"top","length":4,"width":6}}]}',
        "extra key": '{"ops":[{"op":"add","feature":{"kind":"hole","face":"top","d":6,"colour":"red"}}]}',
        "empty resize": '{"ops":[{"op":"resize"}]}',
        "not JSON": 'make it bigger',
    }
    for name, txt in cases.items():
        o, e = parse_ops(txt)
        check(f"parse_ops rejects: {name}", o is None and len(e) > 0, e[0][:70])


def test_analytic():
    pi = math.pi
    exact = {
        "through hole d6 (top)": (T(Hole(face="top", d=6)), pi * 9 * 10),
        "hole d6 through FRONT face": (T(Hole(face="front", d=6)), pi * 9 * 20),
        "blind hole d8 x 4": (T(Hole(face="top", d=8, depth=4)), pi * 16 * 4),
        "counterbore d5, cb 10 x 3": (T(Hole(face="top", d=5, cbore_d=10, cbore_depth=3)), pi * 6.25 * 10 + pi * (25 - 6.25) * 3),
        "countersink d4 -> 8 (90 deg)": (T(Hole(face="top", d=4, csk_d=8)), pi * 4 * 10 + (pi * 2 / 3 * (16 + 8 + 4) - pi * 4 * 2)),
        "bolt circle 4 x d4, PCD 14": (T(BoltPattern(face="top", n=4, pcd=14, d=4)), 4 * pi * 4 * 10),
        "pocket 10 x 6 x 4": (T(PocketRect(face="top", size_u=10, size_v=6, depth=4)), 240),
        "pocket, corner R2": (T(PocketRect(face="top", size_u=10, size_v=6, depth=4, corner_radius=2)), (60 - (4 - pi) * 4) * 4),
        "slot 14 x 6 through": (T(Slot(face="top", length=14, width=6)), ((14 - 6) * 6 + pi * 9) * 10),
        "boss d8 h5 (adds)": (T(Boss(face="top", d=8, height=5)), -pi * 16 * 5),
        "rect boss 6x4 h3 on RIGHT": (T(Boss(face="right", shape="rect", size_u=6, size_v=4, height=3)), -72),
        "shell, open top, t=2 (cavity)": (T(Shell(open_face="top", thickness=2)), 36 * 16 * 8),
    }
    for name, (tree, exp) in exact.items():
        got = removed(tree)
        check(f"analytic volume: {name}", abs(got - exp) / abs(exp) < 1e-9, f"{got:.3f} vs {exp:.3f}")
    for name, tree, exp in (("fillet top edges R1", T(Fillet(selector="top_edges", radius=1)), 120 * (1 - pi / 4)),
                            ("chamfer top edges 1", T(Chamfer(selector="top_edges", distance=1)), 120 * 0.5)):
        got = removed(tree)
        check(f"first-order volume (corner effects ~2%): {name}", abs(got - exp) / exp < 0.03, f"{got:.2f} vs {exp:.2f}")


def test_vs_generator():
    worst, n = 1.0, 0
    for s in range(40):
        b, sp = make_part(np.random.default_rng(s), "prismatic")
        sol = build(from_spec(sp))
        a, c = b.val(), sol.val()
        inter = a.intersect(c).Volume()
        worst = min(worst, inter / (a.Volume() + c.Volume() - inter))
        n += 1
    check(f"executor reproduces {n} independently generated parts (worst volume IoU > 0.99999)", worst > 0.99999, f"{worst:.7f}")


def test_gate():
    s = Session(T())
    cases = [
        ("hole 500 mm off the part (valid solid, no effect)", [AddOp(feature=Hole(face="top", d=6, at=At(u=Ref(ref="min", offset=500))))], "no effect"),
        ("impossible fillet R40 on a 10 mm part (kernel failure)", [AddOp(feature=Fillet(selector="all_edges", radius=40))], "could not build"),
        ("hole larger than the part (everything removed)", [AddOp(feature=Hole(face="top", d=80))], "removed the entire part"),
        ("slot across the whole width (splits the part)", [AddOp(feature=Slot(face="top", length=60, width=6))], "separate solids"),
        ("pocket corner radius too large", [AddOp(feature=PocketRect(face="top", size_u=10, size_v=6, depth=2, corner_radius=4))], "corner_radius"),
        ("unknown feature id", [SetOp(id="hole_9", path="d", value=3)], "no feature"),
        ("unknown path", [AddOp(feature=Hole(face="top", d=6)), SetOp(id="hole_1", path="nope", value=3)], "unknown path"),
        ("invalid value (cbore smaller than hole)", [AddOp(feature=Hole(face="top", d=6)), SetOp(id="hole_1", path="cbore_d", value=4.0)], "cbore"),
    ]
    for name, ops, key in cases:
        v, n = s.current.version, len(s.history)
        p = s.apply(ops)
        check(f"gate rejects: {name}", (not p.ok) and any(key in e for e in p.errors) and s.current.version == v and len(s.history) == n, p.errors[0][:90] if p.errors else "accepted!")
    # atomicity
    v = s.current.version
    p = s.apply([AddOp(feature=Hole(face="top", d=6)), AddOp(feature=Fillet(selector="all_edges", radius=40))])
    check("multi-operation edit is atomic (valid op 1 not kept when op 2 fails)", (not p.ok) and p.failed_op == 1 and s.current.version == v
          and len(s.current.tree.features) == 0, f"failed_op={p.failed_op}")
    # lost effect after a resize
    s2 = Session(T(Hole(face="top", d=4, at=At(u=Ref(ref="min", offset=35)))))
    p = s2.apply([ResizeOp(W=30)])
    check("resize that pushes a feature off the part is rejected", (not p.ok) and any("no longer touch" in e for e in p.errors), p.errors[0][:90] if p.errors else "")
    # warnings (allowed)
    s3 = Session(T(PocketRect(face="top", size_u=10, size_v=6, depth=4)))
    p = s3.apply([AddOp(feature=Hole(face="top", d=4))])
    check("hole through an existing pocket: allowed, with a partial-overlap warning", p.ok and any("partly in material" in w for w in p.warnings), str(p.warnings)[:90])
    s4 = Session(T())
    p = s4.apply([AddOp(feature=PocketRect(face="top", size_u="full", size_v="full", depth=2))])
    check("cutting off the top strip: allowed, warns that overall size changed", p.ok and any("Overall size" in w for w in p.warnings), str(p.warnings)[:90])


def _cut_bbox_center(tree_with, tree_without):
    return tuple(np.round(np.array(build(tree_without).val().cut(build(tree_with).val()).BoundingBox().center.toTuple()), 6))


def test_resize_anchoring():
    h_center = Hole(id="h", face="top", d=4)
    h_min = Hole(id="h", face="top", d=4, at=At(u=Ref(ref="min", offset=35), v=Ref(ref="center")))
    for name, hole, expect_x in (("centre-anchored hole stays centred", h_center, 30.0), ("min-anchored hole keeps its offset", h_min, 35.0)):
        s = Session(T(hole))
        s.apply([ResizeOp(W=60)])
        c = _cut_bbox_center(s.current.tree, FeatureTree(base=s.current.tree.base))
        check(f"resize W 40 -> 60: {name}", abs(c[0] - expect_x) < 1e-6 and abs(c[1] - 10) < 1e-6, f"x={c[0]:.3f}")
    s = Session(T(PocketRect(face="top", size_u=10, size_v="full", depth=3)))
    s.apply([ResizeOp(D=50)])
    p = build(s.current.tree).val().BoundingBox()
    r = removed(s.current.tree)
    check("'full' sized pocket follows a resize of D", abs(r - 10 * 50 * 3) < 1e-6, f"{r:.1f}")


def test_session_history():
    s = Session(T())
    p1 = s.apply([AddOp(feature=Hole(face="top", d=6))])
    p2 = s.apply([AddOp(feature=Fillet(selector="top_edges", radius=2))])
    p3 = s.apply([ResizeOp(W=50)])
    vols = [h.stats["volume"] for h in s.history]
    check("three committed edits -> versions 0..3", [h.version for h in s.history] == [0, 1, 2, 3] and s.current.version == 3)
    check("volumes change as expected (hole, fillet, resize)", abs(vols[0] - 8000) < 1e-6 and vols[1] < vols[0] and vols[2] < vols[1] and vols[3] > vols[2],
          str([round(v) for v in vols]))
    check("undo twice -> version 1; redo -> 2", s.undo().version == 2 and s.undo().version == 1 and s.redo().version == 2)
    pre = len(s.history)
    pr = s.propose([AddOp(feature=Chamfer(selector="bottom_edges", distance=1))])
    check("propose() does not mutate history", pr.ok and len(s.history) == pre and s.current.version == 2)
    s.goto(1)
    s.apply([AddOp(feature=Chamfer(selector="bottom_edges", distance=1))])
    check("editing after undo discards the redo branch", len(s.history) == 3 and s.current.version == 2 and s.redo().version == 2)
    v_a = s.solid(1).val().Volume()
    v_b = build(s.history[1].tree).val().Volume()
    check("rebuilding a stored version is deterministic", v_a == v_b and abs(v_a - vols[1]) < 1e-9)
    check("proposal reports a diff (removed / added volume)", abs(p1.diff["removed"] - math.pi * 9 * 10) < 1e-6 and p1.diff["added"] < 1e-9, f"removed={p1.diff['removed']:.2f}")


def test_edits_and_step():
    s = Session(T(Hole(id="mount", face="top", d=6)))
    v0 = s.current.stats["volume"]
    s.apply([SetOp(id="mount", path="d", value=8.0)])
    check("SetOp changes a parameter: hole d 6 -> 8 removes pi*(16-9)*10 more", abs((v0 - s.current.stats["volume"]) - math.pi * 7 * 10) < 1e-6)
    s.apply([SetOp(id="mount", path="at.u.offset", value=5.0)])
    c = _cut_bbox_center(s.current.tree, FeatureTree(base=s.current.tree.base))
    check("SetOp on a nested path moves the feature (at.u.offset = +5)", abs(c[0] - 25.0) < 1e-6, f"x={c[0]:.2f}")
    s.apply([DeleteOp(id="mount")])
    check("DeleteOp restores the base volume", abs(s.current.stats["volume"] - 8000) < 1e-9)
    s.apply([AddOp(feature=Hole(face="top", d=6, cbore_d=10, cbore_depth=3)), AddOp(feature=Fillet(selector="vertical_edges", radius=2))])
    st = s.current.stats
    check("committed model: single valid solid, STEP round trip < 1e-4", st["n_solids"] == 1 and st["valid"] and st["step_roundtrip_rel_err"] < 1e-4,
          f"rt err {st['step_roundtrip_rel_err']:.1e}")
    path = os.path.join(tempfile.mkdtemp(), "m.step")
    cq.exporters.export(s.solid(), path)
    back = cq.importers.importStep(path).val()
    check("exported STEP re-imports with identical volume", abs(back.Volume() - s.current.stats["volume"]) < 1e-6 and back.isValid())


def test_spec_scenario():
    """'Add a 6 mm mounting hole in the centre and fillet the top edges by 2 mm' on a 40 x 20 x 10 block."""
    s = Session(T())
    ops, err = parse_ops('[{"op":"add","feature":{"kind":"hole","face":"top","d":6}},'
                         '{"op":"add","feature":{"kind":"fillet","selector":"top_edges","radius":2}}]')
    p = s.apply(ops)
    hole = math.pi * 9 * 10
    fil = (120 + math.pi * 6) * 4 * (1 - math.pi / 4)             # top_edges include the hole's rim
    exp = 8000 - hole - fil
    check("spec example: hole + fillet within 1.5% of the analytic estimate", p.ok and abs(p.stats["volume"] - exp) / exp < 0.015,
          f"{p.stats['volume']:.1f} vs {exp:.1f}")
    check("spec example: no warnings", not p.warnings, str(p.warnings))


if __name__ == "__main__":
    test_parse()
    test_analytic()
    test_vs_generator()
    test_gate()
    test_resize_anchoring()
    test_session_history()
    test_edits_and_step()
    test_spec_scenario()
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)
