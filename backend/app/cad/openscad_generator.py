import trimesh
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from typing import Optional, Tuple, List
from pydantic import BaseModel

class ParametricCADGenerator:
    """
    Parametric OpenSCAD synthesizer and 3D CSG compiler.
    Dynamically generates 3D geometry from arbitrary 2D hull contours and detected bores.
    """

    def generate_openscad_code(
        self,
        length: float,
        width: float,
        thickness: float,
        bore_dia: float,
        hull_points: Optional[List[List[float]]] = None,
        corner_radius: float = 12.0,
        hole_dia: float = 6.0
    ) -> str:
        if hull_points and len(hull_points) >= 3:
            pts_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in hull_points])
            scad = f"""// ========================================================
// SpatialForge Autonomous Parametric CAD Model
// Synthesized via Meta Muse Spark Open & SAM 3.1
// ========================================================

$fn = 64;
plate_thick = {thickness:.2f};

polygon_points = [{pts_str}];

linear_extrude(height = plate_thick, center = true, convexity = 10) {{
    polygon(points = polygon_points);
}}
"""
            return scad

        # Fallback to standard parametric CSG
        return f"""// SpatialForge Standard CSG Model
$fn = 64;
base_length = {length:.2f};
base_width = {width:.2f};
plate_thick = {thickness:.2f};
center_bore_dia = {bore_dia:.2f};

difference() {{
    cube([base_length, base_width, plate_thick], center = true);
    cylinder(h = plate_thick + 2, d = center_bore_dia, center = true);
}}
"""

    def compile_mesh_in_memory(
        self,
        length: float,
        width: float,
        thickness: float,
        bore_dia: float,
        hull_points: Optional[List[List[float]]] = None,
        corner_radius: float = 12.0,
        hole_dia: float = 6.0
    ) -> Tuple[trimesh.Trimesh, bytes]:
        """
        In-memory CSG compilation from detected polygon hull.
        """
        if hull_points and len(hull_points) >= 3:
            try:
                # Center the hull points
                pts = np.array(hull_points)
                pts[:, 0] -= np.mean(pts[:, 0])
                pts[:, 1] -= np.mean(pts[:, 1])

                # Scale to target length/width
                span_x = np.max(pts[:, 0]) - np.min(pts[:, 0])
                span_y = np.max(pts[:, 1]) - np.min(pts[:, 1])
                if span_x > 0: pts[:, 0] *= (length / span_x)
                if span_y > 0: pts[:, 1] *= (width / span_y)

                poly = ShapelyPolygon(pts)
                if poly.is_valid:
                    mesh = trimesh.creation.extrude_polygon(poly, height=thickness)
                    mesh.apply_translation([0, 0, -thickness / 2.0])
                    stl_data = mesh.export(file_type='stl')
                    return mesh, stl_data
            except Exception:
                pass

        # Fallback to standard box
        base = trimesh.creation.box(extents=[length, width, thickness])
        bore = trimesh.creation.cylinder(radius=bore_dia / 2.0, height=thickness * 1.5, sections=36)
        try:
            mesh = base.difference([bore])
        except Exception:
            mesh = base

        stl_data = mesh.export(file_type='stl')
        return mesh, stl_data
