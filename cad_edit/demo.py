"""python demo.py [outdir] -- the spec's example session, including a rejected edit, undo/redo and exports."""
import json
import os
import sys

import cadquery as cq

from model import FeatureTree, Block, OpList, parse_ops
from session import Session

out = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(out, exist_ok=True)
s = Session(FeatureTree(base=Block(W=40, D=20, H=10)))
print(f"v0  volume {s.current.stats['volume']:.1f} mm3")


def show(label, p):
    print(f"\n> {label}")
    for m in p.messages:
        print("   op:", m)
    if p.ok:
        d = p.diff
        print(f"   OK   volume {d['volume_before']:.1f} -> {d['volume_after']:.1f}  (removed {d['removed']:.1f}, added {d['added']:.1f})",
              f"  STEP round-trip err {p.stats['step_roundtrip_rel_err']:.0e}")
    for w in p.warnings:
        print("   WARN", w)
    for e in p.errors:
        print("   REJECTED:", e)


ops, _ = parse_ops('[{"op":"add","feature":{"kind":"hole","face":"top","d":6}},'
                   '{"op":"add","feature":{"kind":"fillet","selector":"top_edges","radius":2}}]')
show("Add a 6 mm hole in the centre and fillet the top edges by 2 mm", s.apply(ops))
ops, _ = parse_ops('[{"op":"add","feature":{"kind":"bolt_pattern","face":"top","n":4,"pcd":14,"d":3}}]')
show("Add 4 x M3 clearance holes on a 14 mm PCD", s.apply(ops))
ops, _ = parse_ops('[{"op":"add","feature":{"kind":"hole","face":"top","d":6,"at":{"u":{"ref":"min","offset":500}}}}]')
show("Add a hole 500 mm from the part (valid solid, but a no-op)", s.apply(ops))
ops, _ = parse_ops('[{"op":"resize","W":60}]')
show("Make width 60 mm (centre features stay centred)", s.apply(ops))
print("\nhistory:", [(h.version, h.ops) for h in s.history])
s.undo()
print("after undo -> version", s.current.version, f"volume {s.current.stats['volume']:.1f}")
s.redo()
cq.exporters.export(s.solid(), os.path.join(out, "model.step"))
cq.exporters.export(s.solid(), os.path.join(out, "model.stl"))
with open(os.path.join(out, "model.tree.json"), "w") as fh:
    fh.write(s.current.tree.model_dump_json(indent=2))
print("wrote", sorted(os.listdir(out)))
