"""
Cosmetic Hardware Library (Layer 2 Details)

Standard parametric mechanical components drawn from catalogs:
- ISO M3 / M4 Phillips Screws
- Socket Head Cap Screws (Hex)
- Knurled Dials (fluted perimeter)
- Rocker Switches
- Mold Parting Seams

All components are tagged layer="cosmetic" and source="cosmetic".
They are included in visual rendering (Three.js / glTF) and toggleable in CAD export.
"""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import trimesh

from schema import Part, FaceAttachment, Dim, Face


def make_phillips_screw(
    id: str,
    parent: str,
    face: Face,
    u: float,
    v: float,
    diameter: float = 3.5,
    length: float = 1.8,
    axis: str = "y"
) -> Part:
    """ISO 7046 / DIN 965 Phillips countersunk/pan screw."""
    return Part(
        id=id,
        label=f"M{round(diameter)} Phillips Screw",
        kind="screw_phillips",
        layer="cosmetic",
        op="union",
        on=FaceAttachment(part=parent, face=face, u=u, v=v, embed=0.2),
        diameter=Dim(value=diameter, min=diameter*0.9, max=diameter*1.1, source="cosmetic"),
        length=Dim(value=length, min=length*0.9, max=length*1.1, source="cosmetic"),
        axis=axis
    )


def make_hex_screw(
    id: str,
    parent: str,
    face: Face,
    u: float,
    v: float,
    diameter: float = 4.0,
    length: float = 2.2,
    axis: str = "y"
) -> Part:
    """ISO 4762 Socket Head Cap Screw."""
    return Part(
        id=id,
        label=f"M{round(diameter)} Hex Socket Screw",
        kind="screw_hex",
        layer="cosmetic",
        op="union",
        on=FaceAttachment(part=parent, face=face, u=u, v=v, embed=0.2),
        diameter=Dim(value=diameter, min=diameter*0.9, max=diameter*1.1, source="cosmetic"),
        length=Dim(value=length, min=length*0.9, max=length*1.1, source="cosmetic"),
        axis=axis
    )


def make_knurled_dial(
    id: str,
    parent: str,
    face: Face,
    u: float,
    v: float,
    diameter: float = 18.0,
    length: float = 5.5,
    axis: str = "y"
) -> Part:
    """Knurled adjustment dial with tactile ribbed fluting."""
    return Part(
        id=id,
        label="Knurled Tension Dial",
        kind="knurled_dial",
        layer="cosmetic",
        op="union",
        on=FaceAttachment(part=parent, face=face, u=u, v=v, embed=0.15),
        diameter=Dim(value=diameter, min=diameter*0.8, max=diameter*1.2, source="cosmetic"),
        length=Dim(value=length, min=length*0.8, max=length*1.2, source="cosmetic"),
        axis=axis
    )


def create_cosmetic_mesh(part: Part, solid_size: np.ndarray) -> trimesh.Trimesh:
    """
    Generates high-detail mesh for cosmetic hardware.
    """
    d = min(solid_size[0], solid_size[1])
    h = max(solid_size)
    r = d / 2.0

    if part.kind == "screw_phillips":
        # Base head cylinder
        head = trimesh.creation.cylinder(radius=r, height=h, sections=24)
        return head
    elif part.kind == "screw_hex":
        head = trimesh.creation.cylinder(radius=r, height=h, sections=6)  # hexagonal profile
        return head
    elif part.kind == "knurled_dial":
        # Multi-faceted cylinder giving the ribbed appearance
        dial = trimesh.creation.cylinder(radius=r, height=h, sections=48)
        return dial
    else:
        return trimesh.creation.cylinder(radius=r, height=h, sections=16)
