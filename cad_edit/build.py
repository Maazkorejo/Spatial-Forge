"""
Deterministic executor: FeatureTree -> CadQuery solid. Always rebuilt from scratch (no in-place patching).

For every feature it records a Step with the volume BEFORE/AFTER and an independent analytic `ideal` volume
(cylinder = pi r^2 L, pocket = area x depth, fillet on a convex 90-degree edge = L r^2 (1 - pi/4), ...). The gate
compares the two, which catches edits that are topologically valid but geometrically wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cadquery as cq

from model import (Block, BoltPattern, Boss, Chamfer, Fillet, FeatureTree, Hole, PocketRect, Shell, Slot)

# face -> (u axis, v axis, normal axis, outward sign)
FACE = {"top": (0, 1, 2, 1), "bottom": (0, 1, 2, -1), "front": (0, 2, 1, -1), "back": (0, 2, 1, 1),
        "left": (1, 2, 0, -1), "right": (1, 2, 0, 1)}
FACE_SEL = {"top": ">Z", "bottom": "<Z", "front": "<Y", "back": ">Y", "left": "<X", "right": ">X"}
OVERSHOOT = 1.0


class BuildError(Exception):
    def __init__(self, fid: str, msg: str):
        super().__init__(f"{fid}: {msg}")
        self.fid, self.msg = fid, msg


@dataclass
class Step:
    fid: str
    kind: str
    role: str                         # "cut" | "add" | "edge" | "shell"
    vol_before: float
    vol_after: float
    ideal: Optional[float]            # analytic |volume change| (None if no closed form)

    @property
    def actual(self) -> float:        # positive number = magnitude of change in the expected direction
        return (self.vol_before - self.vol_after) if self.role in ("cut", "shell") else (
            self.vol_after - self.vol_before if self.role == "add" else abs(self.vol_after - self.vol_before))


def _coord(ref, hi: float) -> float:
    return {"min": 0.0, "center": hi / 2, "max": hi}[ref.ref] + ref.offset


class _Face:
    """Geometry of one base-block face and a placement on it."""

    def __init__(self, face: str, dims):
        self.ua, self.va, self.na, self.sg = FACE[face]
        self.dims = dims
        self.ext_u, self.ext_v, self.ext_n = dims[self.ua], dims[self.va], dims[self.na]
        self.n_out = cq.Vector(*[(self.sg if i == self.na else 0.0) for i in range(3)])
        self.u_dir = cq.Vector(*[(1.0 if i == self.ua else 0.0) for i in range(3)])

    def point(self, u: float, v: float, out: float = 0.0) -> cq.Vector:
        p = [0.0, 0.0, 0.0]
        p[self.ua], p[self.va] = u, v
        p[self.na] = (self.ext_n if self.sg > 0 else 0.0) + self.sg * out
        return cq.Vector(*p)

    def plane(self, u, v, out=OVERSHOOT) -> cq.Plane:
        return cq.Plane(origin=self.point(u, v, out), xDir=self.u_dir, normal=self.n_out)


def _wp(shape) -> cq.Workplane:
    return cq.Workplane("XY").newObject([shape])


def _tool_len(depth, ext_n):
    return (ext_n + 2 * OVERSHOOT, ext_n) if depth == "through" else (depth + OVERSHOOT, depth)


# ----------------------------------------------------------------------------- tools
def _hole_tool(f: Hole, dims):
    fc = _Face(f.face, dims)
    u, v = _coord(f.at.u, fc.ext_u), _coord(f.at.v, fc.ext_v)
    tl, lin = _tool_len(f.depth, fc.ext_n)
    r = f.d / 2
    tool = cq.Workplane(fc.plane(u, v)).circle(r).extrude(-tl)
    ideal = math.pi * r * r * lin
    if f.cbore_d:
        R = f.cbore_d / 2
        tool = tool.union(cq.Workplane(fc.plane(u, v)).circle(R).extrude(-(f.cbore_depth + OVERSHOOT)))
        ideal += math.pi * (R * R - r * r) * f.cbore_depth
    if f.csk_d:
        R = f.csk_d / 2
        h = R - r                                              # 90 degree included angle
        cone = cq.Solid.makeCone(R, r, h, fc.point(u, v, 0.0), -fc.n_out)
        cap = cq.Solid.makeCylinder(R, OVERSHOOT, fc.point(u, v, OVERSHOOT), -fc.n_out)
        tool = tool.union(_wp(cone)).union(_wp(cap))
        ideal += math.pi * h / 3 * (R * R + R * r + r * r) - math.pi * r * r * h
    return tool, ideal


def _bolt_tool(f: BoltPattern, dims):
    fc = _Face(f.face, dims)
    u0, v0 = _coord(f.center.u, fc.ext_u), _coord(f.center.v, fc.ext_v)
    tl, lin = _tool_len(f.depth, fc.ext_n)
    cyls = []
    for i in range(f.n):
        a = math.radians(f.start_deg + 360.0 * i / f.n)
        p = fc.point(u0 + f.pcd / 2 * math.cos(a), v0 + f.pcd / 2 * math.sin(a), OVERSHOOT)
        cyls.append(cq.Solid.makeCylinder(f.d / 2, tl, p, -fc.n_out))
    return _wp(cq.Compound.makeCompound(cyls)), f.n * math.pi * (f.d / 2) ** 2 * lin


def _pocket_tool(f: PocketRect, dims):
    fc = _Face(f.face, dims)
    u, v = _coord(f.at.u, fc.ext_u), _coord(f.at.v, fc.ext_v)
    su = fc.ext_u if f.size_u == "full" else f.size_u
    sv = fc.ext_v if f.size_v == "full" else f.size_v
    r = f.corner_radius
    if r > 0 and r > min(su, sv) / 2 - 1e-9:
        raise ValueError("corner_radius must be at most half the smaller pocket side")
    tl, lin = _tool_len(f.depth, fc.ext_n)
    tool = cq.Workplane(fc.plane(u, v)).rect(su, sv).extrude(-tl)
    if r > 0:
        tool = tool.edges(cq.selectors.ParallelDirSelector(fc.n_out)).fillet(r)
    return tool, (su * sv - (4 - math.pi) * r * r) * lin


def _slot_tool(f: Slot, dims):
    fc = _Face(f.face, dims)
    u, v = _coord(f.at.u, fc.ext_u), _coord(f.at.v, fc.ext_v)
    tl, lin = _tool_len(f.depth, fc.ext_n)
    tool = cq.Workplane(fc.plane(u, v)).slot2D(f.length, f.width).extrude(-tl)
    return tool, ((f.length - f.width) * f.width + math.pi * f.width ** 2 / 4) * lin


def _boss_tool(f: Boss, dims):
    fc = _Face(f.face, dims)
    u, v = _coord(f.at.u, fc.ext_u), _coord(f.at.v, fc.ext_v)
    wp = cq.Workplane(fc.plane(u, v, out=-0.5))               # start 0.5 mm inside so the union fuses
    if f.shape == "circ":
        return wp.circle(f.d / 2).extrude(f.height + 0.5), math.pi * (f.d / 2) ** 2 * f.height
    return wp.rect(f.size_u, f.size_v).extrude(f.height + 0.5), f.size_u * f.size_v * f.height


def _edges(solid: cq.Workplane, selector: str) -> cq.Workplane:
    return {"top_edges": lambda s: s.faces(">Z").edges(), "bottom_edges": lambda s: s.faces("<Z").edges(),
            "vertical_edges": lambda s: s.edges("|Z"), "all_edges": lambda s: s.edges()}[selector](solid)


def _vol(wp: cq.Workplane) -> float:
    try:
        return float(wp.val().Volume())
    except Exception:                                         # empty result (everything was removed)
        return 0.0


# ----------------------------------------------------------------------------- build
def build(tree: FeatureTree, trace: Optional[list] = None) -> cq.Workplane:
    tree = tree.with_ids()
    dims = (tree.base.W, tree.base.D, tree.base.H)
    solid = cq.Workplane("XY").box(*dims, centered=False)
    for f in tree.features:
        v0 = _vol(solid)
        ideal, role = None, "cut"
        try:
            if isinstance(f, Hole):
                tool, ideal = _hole_tool(f, dims)
                solid = solid.cut(tool)
            elif isinstance(f, BoltPattern):
                tool, ideal = _bolt_tool(f, dims)
                solid = solid.cut(tool)
            elif isinstance(f, PocketRect):
                tool, ideal = _pocket_tool(f, dims)
                solid = solid.cut(tool)
            elif isinstance(f, Slot):
                tool, ideal = _slot_tool(f, dims)
                solid = solid.cut(tool)
            elif isinstance(f, Boss):
                tool, ideal = _boss_tool(f, dims)
                solid = solid.union(tool)
                role = "add"
            elif isinstance(f, (Fillet, Chamfer)):
                sel = _edges(solid, f.selector)
                L = sum(e.Length() for e in sel.vals())
                if L <= 0:
                    raise ValueError(f"selector '{f.selector}' matched no edges")
                if isinstance(f, Fillet):
                    solid = sel.fillet(f.radius)
                    ideal = L * f.radius ** 2 * (1 - math.pi / 4)
                else:
                    solid = sel.chamfer(f.distance)
                    ideal = L * f.distance ** 2 / 2
                role = "edge"
            elif isinstance(f, Shell):
                solid = solid.faces(FACE_SEL[f.open_face]).shell(-f.thickness)
                role = "shell"
        except BuildError:
            raise
        except Exception as e:                                # OpenCascade failures surface as many exception types
            raise BuildError(f.id, f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
        if trace is not None:
            trace.append(Step(f.id, f.kind, role, v0, _vol(solid), ideal))
    return solid
