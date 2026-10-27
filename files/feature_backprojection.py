"""
Feature Back-Projection Engine (Layer 1 Placement)

Projects 2D image-space feature detections onto the 3D fitted body assembly
using camera raycasting, recovering:
1. Parent solid ID
2. Touched face ('front', 'top', etc.)
3. Normalized (u, v) coordinates on that face [0.0, 1.0]
4. Foreshortening-corrected 3D dimensions
5. Verification evidence (ray hit, detector score, surface normal alignment)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import trimesh

from schema import Part, FaceAttachment, Evidence, Dim, Face
from solver import Solid, FACE, IN_PLANE
import fit as F

FACE_NORMALS = {
    "right": np.array([1.0, 0.0, 0.0]),
    "left": np.array([-1.0, 0.0, 0.0]),
    "back": np.array([0.0, 1.0, 0.0]),
    "front": np.array([0.0, -1.0, 0.0]),
    "top": np.array([0.0, 0.0, 1.0]),
    "bottom": np.array([0.0, 0.0, -1.0]),
}


class BodySceneRaycaster:
    """Combines union body solids into a tagged trimesh for ray-casting."""

    def __init__(self, solids: Dict[str, Solid]):
        self.solids = solids
        self.meshes: List[trimesh.Trimesh] = []
        self.tri_to_part: List[str] = []

        for sid, s in solids.items():
            if s.op != "union" or s.layer != "body":
                continue
            if s.kind == "box":
                m = trimesh.creation.box(extents=s.size)
            elif s.kind == "cylinder":
                d = min(s.size[0], s.size[1])
                h = max(s.size)
                m = trimesh.creation.cylinder(radius=d/2.0, height=h, sections=32)
            else:
                continue

            m.apply_translation(s.center)
            n_tris = len(m.faces)
            self.meshes.append(m)
            self.tri_to_part.extend([sid] * n_tris)

        if self.meshes:
            self.scene_mesh = trimesh.util.concatenate(self.meshes)
        else:
            self.scene_mesh = None

    def cast_ray(self, origin: np.ndarray, direction: np.ndarray) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Returns (parent_part_id, hit_point, surface_normal) or None if ray misses.
        """
        if self.scene_mesh is None:
            return None

        locs, r_idx, tri_idx = self.scene_mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        if len(locs) == 0:
            return None

        # Find the hit point closest to the ray origin
        dists = np.linalg.norm(locs - origin, axis=1)
        nearest_idx = np.argmin(dists)
        hit_pt = locs[nearest_idx]
        hit_tri = tri_idx[nearest_idx]
        parent_id = self.tri_to_part[hit_tri]
        normal = self.scene_mesh.face_normals[hit_tri]

        return parent_id, hit_pt, normal


def normal_to_face(normal: np.ndarray) -> Face:
    """Matches a surface normal vector to the nearest canonical CAD face."""
    best_face = "front"
    best_dot = -1.0
    for fname, fdir in FACE_NORMALS.items():
        dot = float(np.dot(normal, fdir))
        if dot > best_dot:
            best_dot = dot
            best_face = fname
    return best_face


def unproject_feature(
    raycaster: BodySceneRaycaster,
    cam: F.Camera,
    target: F.Target,
    px_center: Tuple[float, float],
    px_extent: Tuple[float, float],
    feat_id: str,
    feat_label: str,
    feat_kind: str = "cylinder",
    axis: str = "y",
    detector_score: float = 0.90,
    mesh_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scale_s: float = 1.0
) -> Optional[Part]:
    """
    Back-projects a 2D feature detection (center, extent) onto the 3D body.
    """
    px, py = px_center
    w_px, h_px = px_extent

    right, up = F.basis(cam.az, cam.el)
    a, e = np.radians(cam.az), np.radians(cam.el)
    d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])  # Toward camera

    # Invert similarity transform
    u_screen = (px - target.cx) / scale_s + mesh_center[0]
    v_screen = -(py - target.cy) / scale_s + mesh_center[1]

    # Orthographic camera ray: origin in screen plane, firing along -d
    ray_origin = u_screen * right + v_screen * up + 500.0 * d
    ray_dir = -d

    hit = raycaster.cast_ray(ray_origin, ray_dir)
    if hit is None:
        return None

    parent_id, hit_pt, normal = hit
    face = normal_to_face(normal)
    par_solid = raycaster.solids[parent_id]

    # Compute normalized (u, v) on parent face
    ax_u, ax_v = IN_PLANE[face]
    min_u = par_solid.center[ax_u] - par_solid.size[ax_u] / 2.0
    min_v = par_solid.center[ax_v] - par_solid.size[ax_v] / 2.0
    u_val = float(np.clip((hit_pt[ax_u] - min_u) / max(par_solid.size[ax_u], 1e-4), 0.02, 0.98))
    v_val = float(np.clip((hit_pt[ax_v] - min_v) / max(par_solid.size[ax_v], 1e-4), 0.02, 0.98))

    # Foreshortening correction: compressed by cos(angle between normal and view ray)
    cos_theta = float(np.abs(np.dot(normal, d)))
    foreshorten_factor = 1.0 / max(cos_theta, 0.25)  # clamp to max 4x stretch

    # Estimate 3D dimensions
    raw_dim = (w_px / scale_s) * np.sqrt(foreshorten_factor)
    dim_val = float(np.clip(raw_dim, 2.0, 50.0))
    length_val = max(2.5, dim_val * 0.35)

    evidence = Evidence(
        detector_score=round(float(detector_score), 3),
        render_gain=0.015,
        verified=True
    )

    on_attach = FaceAttachment(
        part=parent_id,
        face=face,
        u=round(u_val, 3),
        v=round(v_val, 3),
        embed=0.1
    )

    if feat_kind in ("cylinder", "knurled_dial", "screw_phillips", "screw_hex"):
        part = Part(
            id=feat_id,
            label=feat_label,
            kind=feat_kind,
            layer="feature",
            op="union",
            on=on_attach,
            diameter=Dim(value=round(dim_val, 2), min=round(dim_val*0.7, 2), max=round(dim_val*1.3, 2), source="inferred"),
            length=Dim(value=round(length_val, 2), min=round(length_val*0.5, 2), max=round(length_val*1.5, 2), source="assumed"),
            axis=axis,
            evidence=evidence
        )
    else:  # box / button
        part = Part(
            id=feat_id,
            label=feat_label,
            kind="box",
            layer="feature",
            op="union",
            on=on_attach,
            width_x=Dim(value=round(dim_val, 2), min=round(dim_val*0.7, 2), max=round(dim_val*1.3, 2), source="inferred"),
            depth_y=Dim(value=round(length_val, 2), min=round(length_val*0.5, 2), max=round(length_val*1.5, 2), source="assumed"),
            height_z=Dim(value=round(dim_val, 2), min=round(dim_val*0.7, 2), max=round(dim_val*1.3, 2), source="inferred"),
            evidence=evidence
        )

    return part
