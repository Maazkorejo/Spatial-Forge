# cad_edit — deterministic CAD edit engine (parametric feature tree + safety gate + versioned history)

The core under a future "AI copilot": an LLM or UI may only emit validated JSON operations; deterministic code
executes them; a gate verifies the result against independent expectations; every committed edit is a version.

    pip install -r requirements.txt
    python test_edit.py        # 53 checks, ~1 min, CPU only
    python demo.py out         # the spec's example session, writes STEP / STL / tree JSON

## Files
| file | role |
|---|---|
| model.py | Pydantic feature tree + operations (`add`, `set`, `delete`, `resize`); `parse_ops()` for LLM/UI output; `apply_op()` (pure) |
| build.py | tree -> CadQuery solid, rebuilt from scratch; records per-feature volume trace with an analytic expectation |
| gate.py | post-conditions (below) |
| session.py | propose / commit / apply (atomic), undo / redo / goto, diff (removed / added volume) |
| convert.py, gen_parts.py | test support: independently generated parts (copy of drawing_gen/parts.py) -> tree |
| ops_schema.json | JSON schema of the operation list, for constrained decoding |

## Operations (Phase 1)
Features: `hole` (through / blind / counterbore / 90° countersink), `bolt_pattern` (PCD), `pocket_rect` (corner radius),
`slot`, `boss` (rect / circular), `fillet`, `chamfer`, `shell`. Edits: add, set (dotted path), delete, resize W/D/H.
Placement is relational: `(ref = min|center|max, offset)` on each face axis, `size = "full"` follows the part. Edge
selectors are named (`top_edges`, `bottom_edges`, `vertical_edges`, `all_edges`), never indices.
Note: `top_edges` includes the rims of holes/pockets already cut in the top face.

## The gate (what is checked besides isValid())
Rejects: invalid / multiple solids / vanished part; a feature that has no effect, or lost its effect because of
this edit; a cut removing more than its tool volume; geometry outside base+bosses; STEP export->import changing the
volume (> 1e-4). Warns: tool only partly in material; fillet/chamfer volume far from the convex-90° estimate; overall
size changed by a cut. Thresholds are configurable defaults (gate.py), not standards.

## Tested (test_edit.py)
* every op's removed/added volume equals the closed-form value to ~1e-15 (fillet/chamfer: within 3%, corner effect);
* executor reproduces 40 independently generated prismatic parts, worst volume IoU 1.0000000;
* gate rejects: hole off the part, impossible fillet, hole bigger than the part, slot that splits the part, bad
  corner radius, unknown id/path, invalid value; multi-op edits are atomic; resize that strands a feature is rejected;
* resize: centre-anchored feature stays centred, min-anchored keeps its offset, "full" sizes follow;
* history: undo/redo/goto, propose() is side-effect free, editing after undo drops the redo branch, rebuilds deterministic;
* STEP round trip volume error ~1e-15 and valid.

## NOT done / known limits
* No LLM front-end, no natural-language parser, no DFM auditor, no UI. The op JSON is the interface.
* Block base only. Turned parts / arbitrary imported solids need a revolve base or a feature-recognition step.
* Edge selection cannot target "the edges of pocket_2" (needs persistent naming); only the four named selectors.
* No protection against an OpenCascade hang (no timeout / process isolation yet). Production needs it.
* STEP validity is checked by an OpenCascade round trip only. Opening in FreeCAD / SolidWorks / Fusion is NOT tested.
* Units are mm. A model reconstructed from a sketch has relative units until an anchor sets the scale: the copilot
  must refuse or ask before applying mm-valued edits there (enforce at integration).
* Gate effect thresholds (2% no-effect, 90% partial) are heuristics tuned on simple blocks.
* Shell is fragile with fillets / complex faces; failures are caught and reported, not repaired.
