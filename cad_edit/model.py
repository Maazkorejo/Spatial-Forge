"""
Parametric feature tree + edit operations (the ONLY thing an LLM or a UI is allowed to emit).

* The tree is the source of truth. The B-rep is always rebuilt from it (build.py), never patched in place.
* Positions are relational: each feature is placed by (ref, offset) along its face's two axes, where ref is the
  part's min / center / max edge. A centred hole therefore stays centred when the part is resized.
* Edge selection uses NAMED selectors ("top_edges", "vertical_edges", ...), never edge indices, because
  OpenCascade topology ids change after every boolean (the topological-naming problem).
Units: millimetres. Base is a block with its min corner at the origin: x in [0,W], y in [0,D], z in [0,H].

Face-local axes:  top/bottom (u=x, v=y)   front/back (u=x, v=z)   left/right (u=y, v=z)
"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Face = Literal["top", "bottom", "front", "back", "left", "right"]
Selector = Literal["top_edges", "bottom_edges", "vertical_edges", "all_edges"]
Depth = Union[Annotated[float, Field(gt=0, le=2000)], Literal["through"]]
Size = Union[Annotated[float, Field(gt=0, le=2000)], Literal["full"]]
Dim = Annotated[float, Field(gt=0, le=2000)]
ID = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ref(Strict):
    ref: Literal["min", "center", "max"] = "center"
    offset: float = Field(0.0, ge=-2000, le=2000)        # mm along +axis from the reference edge


class At(Strict):
    u: Ref = Field(default_factory=Ref)
    v: Ref = Field(default_factory=Ref)


# --------------------------------------------------------------------------- features
class Hole(Strict):
    kind: Literal["hole"] = "hole"
    id: Optional[ID] = None
    face: Face
    at: At = Field(default_factory=At)
    d: float = Field(gt=0.5, le=500)
    depth: Depth = "through"                              # blind circular pocket == blind hole
    cbore_d: Optional[Dim] = None
    cbore_depth: Optional[Dim] = None
    csk_d: Optional[Dim] = None                            # 90-degree countersink diameter at the face

    @model_validator(mode="after")
    def _v(self):
        if (self.cbore_d is None) != (self.cbore_depth is None):
            raise ValueError("cbore_d and cbore_depth must be given together")
        if self.cbore_d is not None and self.cbore_d <= self.d:
            raise ValueError("cbore_d must be larger than d")
        if self.csk_d is not None and self.csk_d <= self.d:
            raise ValueError("csk_d must be larger than d")
        if self.cbore_d is not None and self.csk_d is not None:
            raise ValueError("use either a counterbore or a countersink, not both")
        if isinstance(self.depth, float) and self.cbore_depth is not None and self.cbore_depth >= self.depth:
            raise ValueError("cbore_depth must be smaller than depth")
        return self


class BoltPattern(Strict):
    kind: Literal["bolt_pattern"] = "bolt_pattern"
    id: Optional[ID] = None
    face: Face
    center: At = Field(default_factory=At)
    n: int = Field(ge=2, le=24)
    pcd: Dim
    d: float = Field(gt=0.5, le=200)
    depth: Depth = "through"
    start_deg: float = Field(0.0, ge=-360, le=360)

    @model_validator(mode="after")
    def _v(self):
        if self.pcd <= self.d:
            raise ValueError("pcd must be larger than the hole diameter")
        return self


class PocketRect(Strict):
    kind: Literal["pocket_rect"] = "pocket_rect"
    id: Optional[ID] = None
    face: Face
    at: At = Field(default_factory=At)
    size_u: Size
    size_v: Size
    depth: Depth
    corner_radius: float = Field(0.0, ge=0, le=200)


class Slot(Strict):
    kind: Literal["slot"] = "slot"
    id: Optional[ID] = None
    face: Face
    at: At = Field(default_factory=At)
    length: Dim                                           # overall length along the face u axis
    width: Dim
    depth: Depth = "through"

    @model_validator(mode="after")
    def _v(self):
        if self.length <= self.width:
            raise ValueError("slot length must exceed its width")
        return self


class Boss(Strict):
    kind: Literal["boss"] = "boss"
    id: Optional[ID] = None
    face: Face
    at: At = Field(default_factory=At)
    shape: Literal["rect", "circ"] = "circ"
    d: Optional[Dim] = None
    size_u: Optional[Dim] = None
    size_v: Optional[Dim] = None
    height: Dim

    @model_validator(mode="after")
    def _v(self):
        if self.shape == "circ" and (self.d is None or self.size_u or self.size_v):
            raise ValueError("circular boss needs 'd' only")
        if self.shape == "rect" and (self.size_u is None or self.size_v is None or self.d):
            raise ValueError("rect boss needs size_u and size_v only")
        return self


class Fillet(Strict):
    kind: Literal["fillet"] = "fillet"
    id: Optional[ID] = None
    selector: Selector
    radius: float = Field(gt=0.1, le=100)


class Chamfer(Strict):
    kind: Literal["chamfer"] = "chamfer"
    id: Optional[ID] = None
    selector: Selector
    distance: float = Field(gt=0.1, le=100)


class Shell(Strict):
    kind: Literal["shell"] = "shell"
    id: Optional[ID] = None
    open_face: Face
    thickness: float = Field(gt=0.2, le=100)


Feature = Annotated[Union[Hole, BoltPattern, PocketRect, Slot, Boss, Fillet, Chamfer, Shell], Field(discriminator="kind")]


class Block(Strict):
    W: Dim
    D: Dim
    H: Dim


class FeatureTree(Strict):
    base: Block
    features: list[Feature] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _ids(self):
        ids = [f.id for f in self.features if f.id]
        if len(ids) != len(set(ids)):
            raise ValueError("feature ids must be unique")
        return self

    def with_ids(self) -> "FeatureTree":
        """Fill in missing ids deterministically: <kind>_<n>."""
        t = self.model_copy(deep=True)
        used = {f.id for f in t.features if f.id}
        for f in t.features:
            if not f.id:
                n = 1
                while f"{f.kind}_{n}" in used:
                    n += 1
                f.id = f"{f.kind}_{n}"
                used.add(f.id)
        return t

    def get(self, fid: str):
        return next((f for f in self.features if f.id == fid), None)


# --------------------------------------------------------------------------- operations
class AddOp(Strict):
    op: Literal["add"] = "add"
    feature: Feature


class SetOp(Strict):
    op: Literal["set"] = "set"
    id: ID
    path: str = Field(description="field of the feature, dotted for nested, e.g. 'd', 'depth', 'at.u.offset'")
    value: Union[float, str, bool, None]


class DeleteOp(Strict):
    op: Literal["delete"] = "delete"
    id: ID


class ResizeOp(Strict):
    op: Literal["resize"] = "resize"
    W: Optional[Dim] = None
    D: Optional[Dim] = None
    H: Optional[Dim] = None

    @model_validator(mode="after")
    def _v(self):
        if self.W is None and self.D is None and self.H is None:
            raise ValueError("resize needs at least one of W, D, H")
        return self


Op = Annotated[Union[AddOp, SetOp, DeleteOp, ResizeOp], Field(discriminator="op")]


class OpList(Strict):
    ops: list[Op] = Field(min_length=1, max_length=8)


def parse_ops(text: str) -> tuple[Optional[list], list[str]]:
    """LLM / UI output -> validated ops. Errors are phrased so they can be fed back to the model."""
    import re
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(t)
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON: {e}"]
    if isinstance(data, list):
        data = {"ops": data}
    try:
        return OpList.model_validate(data).ops, []
    except ValidationError as e:
        return None, [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()][:10]


# --------------------------------------------------------------------------- applying ops to a tree (pure)
class OpError(Exception):
    pass


def _set_path(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            raise OpError(f"unknown path '{path}'")
        cur = cur[k]
    if not isinstance(cur, dict) or keys[-1] not in cur:
        raise OpError(f"unknown path '{path}'")
    if keys[-1] in ("kind", "id"):
        raise OpError("'kind' and 'id' cannot be changed")
    cur[keys[-1]] = value


def apply_op(tree: FeatureTree, op) -> FeatureTree:
    """Returns a NEW tree. Raises OpError for malformed edits (unknown id/path, invalid value)."""
    t = tree.with_ids().model_copy(deep=True)
    if isinstance(op, AddOp):
        t.features.append(op.feature.model_copy(deep=True))
        return t.with_ids()
    if isinstance(op, DeleteOp):
        if t.get(op.id) is None:
            raise OpError(f"no feature with id '{op.id}'")
        t.features = [f for f in t.features if f.id != op.id]
        return t
    if isinstance(op, ResizeOp):
        for k in ("W", "D", "H"):
            v = getattr(op, k)
            if v is not None:
                setattr(t.base, k, v)
        return t
    if isinstance(op, SetOp):
        f = t.get(op.id)
        if f is None:
            raise OpError(f"no feature with id '{op.id}'")
        d = f.model_dump()
        _set_path(d, op.path, op.value)
        try:
            new = type(f).model_validate(d)
        except ValidationError as e:
            raise OpError("; ".join((f"{'.'.join(str(x) for x in err['loc'])}: " if err["loc"] else "") + err["msg"].replace("Value error, ", "")
                                    for err in e.errors()))
        t.features = [new if x.id == op.id else x for x in t.features]
        return t
    raise OpError("unsupported operation")


def describe(op, tree: Optional[FeatureTree] = None) -> str:
    if isinstance(op, AddOp):
        f = op.feature
        if f.kind == "hole":
            extra = f", cbore d={f.cbore_d}x{f.cbore_depth}" if f.cbore_d else (f", countersink d={f.csk_d}" if f.csk_d else "")
            return f"Add hole d={f.d} mm ({'through' if f.depth == 'through' else f'{f.depth} mm deep'}{extra}) on {f.face}"
        if f.kind == "bolt_pattern":
            return f"Add {f.n}x d={f.d} mm bolt circle, PCD {f.pcd} mm, on {f.face}"
        if f.kind == "pocket_rect":
            return f"Add rectangular pocket {f.size_u}x{f.size_v} mm, depth {f.depth}, on {f.face}"
        if f.kind == "slot":
            return f"Add slot {f.length}x{f.width} mm, depth {f.depth}, on {f.face}"
        if f.kind == "boss":
            return f"Add {f.shape} boss, height {f.height} mm, on {f.face}"
        if f.kind == "fillet":
            return f"Fillet {f.selector} R{f.radius} mm"
        if f.kind == "chamfer":
            return f"Chamfer {f.selector} {f.distance} mm"
        if f.kind == "shell":
            return f"Shell wall {f.thickness} mm, open {f.open_face}"
    if isinstance(op, SetOp):
        return f"Set {op.id}.{op.path} = {op.value}"
    if isinstance(op, DeleteOp):
        return f"Delete {op.id}"
    if isinstance(op, ResizeOp):
        return "Resize " + ", ".join(f"{k}={getattr(op, k)} mm" for k in ("W", "D", "H") if getattr(op, k) is not None)
    return str(op)
