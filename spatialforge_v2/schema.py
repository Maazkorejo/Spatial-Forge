"""
SpatialForge relational CSG DSL (v2: + rounded_box, extrusion, features)

Design rules
------------
* The VLM decides DISCRETE things: which parts exist, primitive type, what attaches to
  what, which face, alignment, boolean op.
* The VLM gives CONTINUOUS things only as (value, min, max, source). The fitter in
  fit.py may move `value` inside [min, max]. Nothing here uses absolute coordinates, so
  parts cannot float or interpenetrate when a dimension changes.
* Units are arbitrary "u". The VLM is told to make the longest overall extent ~100u.
  Real millimetres come only from an `anchor` (a printed dimension or a user input).

Coordinate convention: x = right, y = back (away from a viewer facing the front), z = up.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Source = Literal["measured", "inferred", "assumed"]
Face = Literal["top", "bottom", "left", "right", "front", "back"]
Align = Literal["center", "min", "max"]
Kind = Literal["box", "cylinder", "sphere", "rounded_box", "extrusion"]
FeatureKind = Literal["button", "dial", "screw", "hole", "slot", "switch"]
ROUND_FEATURES = ("button", "dial", "screw", "hole")
RECT_FEATURES = ("slot", "switch")

ALL_DIMS = ("width_x", "depth_y", "height_z", "diameter", "length")
DIM_FIELDS = {
    "box": ("width_x", "depth_y", "height_z"),
    "rounded_box": ("width_x", "depth_y", "height_z"),
    "extrusion": ("width_x", "depth_y", "height_z"),
    "cylinder": ("diameter", "length"),
    "sphere": ("diameter",),
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dim(Strict):
    """A continuous quantity with an explicit uncertainty range and provenance."""
    value: float = Field(gt=0, le=1000, description="Best guess in u")
    min: float = Field(gt=0, le=1000, description="Lowest plausible value in u")
    max: float = Field(gt=0, le=1000, description="Highest plausible value in u")
    source: Source = Field("inferred", description=(
        "measured = a printed dimension was read from the image; "
        "inferred = proportion visible in the image; assumed = hidden/unknowable"))

    @model_validator(mode="after")
    def _range(self):
        if self.min > self.max:
            raise ValueError("min must be <= max")
        if not (self.min - 1e-9 <= self.value <= self.max + 1e-9):
            raise ValueError(f"value {self.value} is outside [{self.min}, {self.max}]")
        return self


class Placement(Strict):
    """Attach this part to a face of its parent. Offsets are in u along +x/+y/+z."""
    parent: str
    face: Face = Field(description="Which face of the PARENT this part touches")
    align_x: Align = "center"
    align_y: Align = "center"
    align_z: Align = "center"
    offset_x: float = Field(0.0, ge=-500, le=500)
    offset_y: float = Field(0.0, ge=-500, le=500)
    offset_z: float = Field(0.0, ge=-500, le=500)
    embed: float = Field(0.0, ge=0.0, le=1.0, description=(
        "Fraction of this part's size along the face normal that sinks INTO the parent. "
        "0 = just touching, 0.5 = half sunk, ~0.9 for a pocket/cut"))
    # align/offset on the axis normal to `face` are ignored; `embed` controls that axis.


def _seg_intersect(a, b, c, d) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    return (ccw(a, c, d) * ccw(b, c, d) < 0) and (ccw(a, b, c) * ccw(a, b, d) < 0)


def _check_profile(pid: str, pts: list[list[float]]):
    if not (3 <= len(pts) <= 24):
        raise ValueError(f"part '{pid}': profile needs 3-24 points")
    for p in pts:
        if len(p) != 2 or not all(0.0 <= v <= 1.0 for v in p):
            raise ValueError(f"part '{pid}': profile points must be [x, z] within 0..1")
    area = 0.0
    for i in range(len(pts)):
        x1, z1 = pts[i]
        x2, z2 = pts[(i + 1) % len(pts)]
        area += x1 * z2 - x2 * z1
    if abs(area) < 1e-3:
        raise ValueError(f"part '{pid}': profile has ~zero area")
    if area < 0:
        raise ValueError(f"part '{pid}': profile must be counter-clockwise (reverse the point order)")
    n = len(pts)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _seg_intersect(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                raise ValueError(f"part '{pid}': profile edges cross (must be a simple polygon)")


class Part(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str = Field(max_length=60)
    kind: Kind
    op: Literal["union", "difference"] = "union"
    cuts: list[str] = Field(default_factory=list, description="For op=difference: ids it subtracts from")
    placement: Optional[Placement] = Field(None, description="Null only for the single root part")
    width_x: Optional[Dim] = None      # box
    depth_y: Optional[Dim] = None      # box
    height_z: Optional[Dim] = None     # box
    diameter: Optional[Dim] = None     # cylinder, sphere
    length: Optional[Dim] = None       # cylinder (along `axis`)
    axis: Optional[Literal["x", "y", "z"]] = None   # cylinder only
    fillet: float = Field(0.0, ge=0, le=50, description="Edge fillet radius in u (rounded_box: real geometry)")
    profile: Optional[list[list[float]]] = Field(None, description=(
        "extrusion only: simple polygon in the X-Z plane, counter-clockwise, 3-24 points, each [x, z] in 0..1 "
        "of the part's width_x/height_z box. It is extruded along y by depth_y."))

    @model_validator(mode="after")
    def _by_kind(self):
        need = DIM_FIELDS[self.kind]
        for f in ALL_DIMS:
            present = getattr(self, f) is not None
            if f in need and not present:
                raise ValueError(f"part '{self.id}': kind={self.kind} requires '{f}'")
            if f not in need and present:
                raise ValueError(f"part '{self.id}': kind={self.kind} must not set '{f}'")
        if (self.kind == "extrusion") != (self.profile is not None):
            raise ValueError(f"part '{self.id}': 'profile' is required for extrusions and forbidden otherwise")
        if self.profile is not None:
            _check_profile(self.id, self.profile)
        if self.kind == "rounded_box" and self.fillet <= 0:
            raise ValueError(f"part '{self.id}': rounded_box needs fillet > 0")
        if (self.kind == "cylinder") != (self.axis is not None):
            raise ValueError(f"part '{self.id}': 'axis' is required for cylinders and forbidden otherwise")
        if self.op == "difference" and not self.cuts:
            raise ValueError(f"part '{self.id}': op=difference needs a non-empty 'cuts' list")
        if self.op == "union" and self.cuts:
            raise ValueError(f"part '{self.id}': 'cuts' is only valid with op=difference")
        return self

    def dims(self) -> dict[str, Dim]:
        return {f: getattr(self, f) for f in DIM_FIELDS[self.kind]}


class FeatureSite(Strict):
    """Where a feature sits: a face of a part, in normalised face coordinates (0..1).
    Faces use tangents (t1, t2): normal x -> (y, z); normal y -> (x, z); normal z -> (x, y)."""
    part: str
    face: Face
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)


class Feature(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    kind: FeatureKind
    on: FeatureSite
    diameter: Optional[Dim] = None     # button, dial, screw, hole
    size_u: Optional[Dim] = None       # slot, switch (along t1)
    size_v: Optional[Dim] = None       # slot, switch (along t2)
    depth: Dim                          # protrusion (button/dial/screw/switch) or cut depth (hole/slot)
    layer: Literal["feature", "cosmetic"] = "feature"
    detector_score: Optional[float] = Field(None, ge=0, le=1)
    incidence_cos: Optional[float] = Field(None, ge=0, le=1, description="|normal . view|; low = grazing, unreliable")
    verified: bool = False

    @model_validator(mode="after")
    def _by_kind(self):
        rnd = self.kind in ROUND_FEATURES
        if rnd and (self.diameter is None or self.size_u is not None or self.size_v is not None):
            raise ValueError(f"feature '{self.id}': {self.kind} needs 'diameter' only")
        if not rnd and (self.size_u is None or self.size_v is None or self.diameter is not None):
            raise ValueError(f"feature '{self.id}': {self.kind} needs 'size_u' and 'size_v' only")
        return self

    def dims(self) -> dict:
        d = {"depth": self.depth}
        if self.kind in ROUND_FEATURES:
            d["diameter"] = self.diameter
        else:
            d["size_u"], d["size_v"] = self.size_u, self.size_v
        return d


class ViewHint(Strict):
    projection: Literal["isometric", "perspective", "orthographic_front",
                        "orthographic_side", "orthographic_top", "unknown"] = "unknown"
    azimuth_deg: Optional[float] = Field(None, ge=-180, le=180, description="0 = looking at the front")
    elevation_deg: Optional[float] = Field(None, ge=-90, le=90, description="0 = eye level, 90 = from above")


class Anchor(Strict):
    part: str
    dim: str
    mm: float = Field(gt=0)


class Assembly(Strict):
    # Field ORDER matters for autoregressive decoding: observations act as a scratchpad
    # that the structure is then conditioned on.
    observations: list[str] = Field(max_length=8, description=(
        "<=8 short factual notes about what is visible: object, viewing angle, main components"))
    object_class: str = Field(max_length=60)
    view_hint: ViewHint = Field(default_factory=ViewHint)
    parts: list[Part] = Field(min_length=1, max_length=24)
    features: list[Feature] = Field(default_factory=list, max_length=40)
    anchor: Optional[Anchor] = Field(None, description="Only if a dimension is PRINTED in the image")
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _graph(self):
        ids = [p.id for p in self.parts]
        if len(set(ids)) != len(ids):
            raise ValueError("part ids must be unique")
        by_id = {p.id: p for p in self.parts}

        roots = [p for p in self.parts if p.placement is None]
        if len(roots) != 1:
            raise ValueError(f"exactly one root part (placement=null) required, found {len(roots)}")
        if roots[0].op != "union":
            raise ValueError("the root part must have op=union")

        for p in self.parts:
            if p.placement is not None:
                par = p.placement.parent
                if par not in by_id:
                    raise ValueError(f"part '{p.id}': unknown parent '{par}'")
                if by_id[par].op == "difference":
                    raise ValueError(f"part '{p.id}': cannot attach to a cut part '{par}'")
            for c in p.cuts:
                if c not in by_id or by_id[c].op != "union":
                    raise ValueError(f"part '{p.id}': cuts target '{c}' must be an existing union part")

        # cycle check
        for p in self.parts:
            seen, cur = set(), p
            while cur.placement is not None:
                if cur.id in seen:
                    raise ValueError(f"placement cycle through '{cur.id}'")
                seen.add(cur.id)
                cur = by_id[cur.placement.parent]

        fids = [f.id for f in self.features]
        if len(set(fids)) != len(fids) or set(fids) & set(ids):
            raise ValueError("feature ids must be unique and different from part ids")
        for f in self.features:
            if f.on.part not in by_id or by_id[f.on.part].op != "union":
                raise ValueError(f"feature '{f.id}': 'on.part' must be an existing union part")

        if self.anchor is not None:
            a = self.anchor
            if a.part not in by_id or a.dim not in DIM_FIELDS[by_id[a.part].kind]:
                raise ValueError("anchor refers to an unknown part/dimension")
        return self

    # ---- helpers -------------------------------------------------------------------
    def topo(self) -> list[Part]:
        by_id = {p.id: p for p in self.parts}
        out, done = [], set()

        def visit(p: Part):
            if p.id in done:
                return
            if p.placement is not None:
                visit(by_id[p.placement.parent])
            done.add(p.id)
            out.append(p)

        for p in self.parts:
            visit(p)
        return out

    def dim_slots(self, include_cuts: bool = False) -> list[tuple[str, str, Dim]]:
        """Continuous parameters. Cut parts are excluded by default: a silhouette cannot
        constrain them, so letting an optimizer move them would just add noise."""
        slots = []
        for p in self.parts:
            if p.op == "difference" and not include_cuts:
                continue
            for f, d in p.dims().items():
                slots.append((p.id, f, d))
        return slots


# ---- parsing / schema export ---------------------------------------------------------
def parse_assembly(text: str) -> tuple[Optional[Assembly], list[str]]:
    """Returns (assembly, errors). Errors are phrased so they can be fed back to the VLM."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        data = json.loads(t)
    except json.JSONDecodeError as e:
        return None, [f"Output is not valid JSON: {e}"]
    try:
        return Assembly.model_validate(data), []
    except ValidationError as e:
        errs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "(root)"
            errs.append(f"{loc}: {err['msg']}")
        return None, errs[:12]


def inline_refs(schema: dict) -> dict:
    """Some constrained-decoding backends dislike $ref/$defs; inline them."""
    defs = schema.get("$defs", {})

    def walk(n):
        if isinstance(n, dict):
            if "$ref" in n:
                return walk(defs[n["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in n.items() if k != "$defs"}
        if isinstance(n, list):
            return [walk(x) for x in n]
        return n

    return walk(schema)


def json_schema(flat: bool = True, for_vlm: bool = True) -> dict:
    """for_vlm=True drops `features`: small features are detected + back-projected, never hallucinated."""
    s = Assembly.model_json_schema()
    s = inline_refs(s) if flat else s
    if for_vlm:
        root = s if flat else s["$defs"]["Assembly"] if "Assembly" in s.get("$defs", {}) else s
        root.get("properties", {}).pop("features", None)
    return s
