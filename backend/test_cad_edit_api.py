import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAD_EDIT_DIR = os.path.join(BASE_DIR, "cad_edit")
if CAD_EDIT_DIR not in sys.path:
    sys.path.insert(0, CAD_EDIT_DIR)

from model import FeatureTree, Block, parse_ops
from session import Session
import cadquery as cq

def test_wiring():
    # 1. Initialize session with 40x20x10 block
    tree = FeatureTree(base=Block(W=40, D=20, H=10))
    s = Session(tree)
    print("Initial volume:", s.current.stats["volume"])
    assert abs(s.current.stats["volume"] - 8000.0) < 1e-6

    # 2. Propose an operation: 6mm hole in center + 2mm fillet on top edges
    json_ops = '[{"op":"add","feature":{"kind":"hole","face":"top","d":6}}, {"op":"add","feature":{"kind":"fillet","selector":"top_edges","radius":2}}]'
    ops, errs = parse_ops(json_ops)
    assert not errs, f"Parse error: {errs}"
    
    p = s.propose(ops)
    print("Proposal ok:", p.ok)
    print("Messages:", p.messages)
    print("Diff:", p.diff)
    assert p.ok
    assert p.diff["removed"] > 280.0

    # 3. Apply the proposal
    snap = s.commit(p)
    print("Committed version:", snap.version)
    assert snap.version == 1

    # 4. Tessellate solid for Three.js
    solid = s.solid()
    verts, triangles = solid.val().tessellate(0.1, 0.2)
    print(f"Tessellated mesh: {len(verts)} vertices, {len(triangles)} triangles")
    assert len(verts) > 0
    assert len(triangles) > 0

    # 5. Test undo
    s.undo()
    print("After undo, version:", s.current.version)
    assert s.current.version == 0
    assert abs(s.current.stats["volume"] - 8000.0) < 1e-6

    # 6. Test redo
    s.redo()
    print("After redo, version:", s.current.version)
    assert s.current.version == 1

    print("ALL CAD_EDIT WIRING TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_wiring()
