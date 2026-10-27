"""
SpatialForge relational CSG DSL  (Part 1 of 2: schema)

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

Source = Literal["measured", "inferred", "assumed", "cosmetic"]
Face = Literal["top", "bottom", "left", "right", "front", "back"]
Align = Literal["center", "min", "max"]
Kind = Literal["box", "cylinder", "sphere", "screw_phillips", "screw_hex", "knurled_dial", "rocker_switch"]
LayerType = Literal["body", "feature", "cosmetic"]

ALL_DIMS = ("width_x", "depth_y", "height_z", "diameter", "length")
DIM_FIELDS = {
    "box": ("width_x", "depth_y", "height_z"),
    "cylinder": ("diameter", "length"),
    "sphere": ("diameter",),
    "screw_phillips": ("diameter", "length"),
    "screw_hex": ("diameter", "length"),
    "knurled_dial": ("diameter", "length"),
    "rocker_switch": ("width_x", "depth_y", "height_z"),
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
        "inferred = proportion visible in the image; assumed = hidden/unknowable; cosmetic = decorative"))

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
    embed: float = Field(0.0, ge=-2.0, le=2.0, description=(
        "Fraction of this part's size along the face normal that sinks INTO the parent. "
        "0 = just touching, 0.5 = half sunk, ~0.9 for a pocket/cut, negative for protruding"))
    # align/offset on the axis normal to `face` are ignored; `embed` controls that axis.


class FaceAttachment(Strict):
    """Normalized coordinate attachment on a parent solid's face (for features & cosmetics)."""
    part: str = Field(description="Parent part id")
    face: Face = Field(description="Which face of the PARENT this feature touches")
    u: float = Field(0.5, ge=0.0, le=1.0, description="Normalized horizontal coordinate [0, 1]")
    v: float = Field(0.5, ge=0.0, le=1.0, description="Normalized vertical coordinate [0, 1]")
    embed: float = Field(0.0, ge=-2.0, le=2.0, description="Fraction of size sinking into parent")


class Evidence(Strict):
    """Verification metrics for detected features."""
    detector_score: float = Field(1.0, ge=0.0, le=1.0, description="Confidence of 2D feature detector")
    render_gain: float = Field(0.0, description="Improvement in local render agreement")
    verified: bool = Field(True, description="True if passed raycast and validation checks")


class Part(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str = Field(max_length=60)
    kind: Kind
    layer: LayerType = Field("body", description="Layer hierarchy: body (macro), feature (functional), cosmetic")
    op: Literal["union", "difference"] = "union"
    cuts: list[str] = Field(default_factory=list, description="For op=difference: ids it subtracts from")
    placement: Optional[Placement] = Field(None, description="Placement relative to parent")
    on: Optional[FaceAttachment] = Field(None, description="Normalized placement on parent face (features/cosmetics)")
    evidence: Optional[Evidence] = Field(None, description="Detection & validation evidence")
    width_x: Optional[Dim] = None      # box, rocker_switch
    depth_y: Optional[Dim] = None      # box, rocker_switch
    height_z: Optional[Dim] = None     # box, rocker_switch
    diameter: Optional[Dim] = None     # cylinder, sphere, screws, dials
    length: Optional[Dim] = None       # cylinder, screws, dials (along `axis`)
    axis: Optional[Literal["x", "y", "z"]] = None   # required for cylinders, screws, dials
    fillet: float = Field(0.0, ge=0, le=50, description="Edge fillet radius in u (CAD export only)")

    @model_validator(mode="after")
    def _by_kind(self):
        need = DIM_FIELDS[self.kind]
        for f in ALL_DIMS:
            present = getattr(self, f) is not None
            if f in need and not present:
                raise ValueError(f"part '{self.id}': kind={self.kind} requires '{f}'")
            if f not in need and present:
                raise ValueError(f"part '{self.id}': kind={self.kind} must not set '{f}'")
        needs_axis = self.kind in ("cylinder", "screw_phillips", "screw_hex", "knurled_dial")
        if needs_axis != (self.axis is not None):
            raise ValueError(f"part '{self.id}': 'axis' is required for {self.kind} and forbidden otherwise")
        if self.placement is not None and self.on is not None:
            raise ValueError(f"part '{self.id}': cannot set both 'placement' and 'on'")
        if self.op == "difference" and not self.cuts:
            raise ValueError(f"part '{self.id}': op=difference needs a non-empty 'cuts' list")
        if self.op == "union" and self.cuts:
            raise ValueError(f"part '{self.id}': 'cuts' is only valid with op=difference")
        return self

    def dims(self) -> dict[str, Dim]:
        return {f: getattr(self, f) for f in DIM_FIELDS[self.kind]}

    def parent_id(self) -> Optional[str]:
        if self.placement is not None:
            return self.placement.parent
        if self.on is not None:
            return self.on.part
        return None


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
    parts: list[Part] = Field(min_length=1, max_length=48)
    features: list[Part] = Field(default_factory=list, description="Layer 1: Functional features (buttons, dials, slots)")
    cosmetics: list[Part] = Field(default_factory=list, description="Layer 2: Decorative hardware (screws, seams, text)")
    anchor: Optional[Anchor] = Field(None, description="Only if a dimension is PRINTED in the image")
    confidence: float = Field(ge=0, le=1)

    def all_parts(self) -> list[Part]:
        return self.parts + self.features + self.cosmetics

    @model_validator(mode="after")
    def _graph(self):
        all_p = self.all_parts()
        ids = [p.id for p in all_p]
        if len(set(ids)) != len(ids):
            raise ValueError("part ids must be unique across all layers")
        by_id = {p.id: p for p in all_p}

        roots = [p for p in self.parts if p.parent_id() is None]
        if len(roots) != 1:
            raise ValueError(f"exactly one root part required, found {len(roots)}")
        if roots[0].op != "union":
            raise ValueError("the root part must have op=union")

        for p in all_p:
            par_id = p.parent_id()
            if par_id is not None:
                if par_id not in by_id:
                    raise ValueError(f"part '{p.id}': unknown parent '{par_id}'")
                if by_id[par_id].op == "difference":
                    raise ValueError(f"part '{p.id}': cannot attach to a cut part '{par_id}'")
            for c in p.cuts:
                if c not in by_id or by_id[c].op != "union":
                    raise ValueError(f"part '{p.id}': cuts target '{c}' must be an existing union part")

        # cycle check
        for p in all_p:
            seen, cur = set(), p
            while cur.parent_id() is not None:
                if cur.id in seen:
                    raise ValueError(f"placement cycle through '{cur.id}'")
                seen.add(cur.id)
                cur = by_id[cur.parent_id()]

        if self.anchor is not None:
            a = self.anchor
            if a.part not in by_id or a.dim not in DIM_FIELDS[by_id[a.part].kind]:
                raise ValueError("anchor refers to an unknown part/dimension")
        return self

    # ---- helpers -------------------------------------------------------------------
    def topo(self) -> list[Part]:
        all_p = self.all_parts()
        by_id = {p.id: p for p in all_p}
        out, done = [], set()

        def visit(p: Part):
            if p.id in done:
                return
            par_id = p.parent_id()
            if par_id is not None and par_id in by_id:
                visit(by_id[par_id])
            done.add(p.id)
            out.append(p)

        for p in all_p:
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


def json_schema(flat: bool = True) -> dict:
    s = Assembly.model_json_schema()
    return inline_refs(s) if flat else s
