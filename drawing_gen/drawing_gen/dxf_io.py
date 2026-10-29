"""DXF writer / reader (ezdxf). The writer drops the ground-truth `role`; native DIMENSION entities are written
when the IR still has dim_* entities (noise.py can explode them into lines + text instead)."""
from __future__ import annotations

import ezdxf

LAYERS = {"VISIBLE": ("CONTINUOUS", 7), "HIDDEN": ("DASHED", 8), "CENTER": ("CENTER", 1), "DIM": ("CONTINUOUS", 3),
          "TITLE": ("CONTINUOUS", 5)}


def write_dxf(drawing: dict, path: str, units_header: bool = True) -> str:
    doc = ezdxf.new("R2010", setup=True)
    doc.units = 4 if units_header else 0                    # 4 = millimetres, 0 = unitless
    for name, (lt, col) in LAYERS.items():
        doc.layers.add(name, color=col, linetype=lt)
    msp = doc.modelspace()
    ov = {"dimtxt": TEXT_H, "dimasz": 2.5, "dimexe": 1.5, "dimexo": 1.0, "dimgap": 1.0,
          "dimlfac": 1.0, "dimdec": 1, "dimzin": 8}      # ezdxf's default style assumes metres (x100) and 2 decimals
    for e in drawing["ents"]:
        a = {"layer": e["layer"], "linetype": e["ltype"]}
        g = e["g"]
        if e["kind"] == "line":
            msp.add_line(g["p"][:2], g["p"][2:], dxfattribs=a)
        elif e["kind"] == "circle":
            msp.add_circle(g["c"], g["r"], dxfattribs=a)
        elif e["kind"] == "arc":
            msp.add_arc(g["c"], g["r"], g["a0"], g["a1"], dxfattribs=a)
        elif e["kind"] == "text":
            msp.add_text(g["text"], height=g["height"], dxfattribs={"layer": e["layer"], "insert": g["at"]})
        elif e["kind"] == "dim_linear":
            msp.add_linear_dim(base=g["base"], p1=g["p1"], p2=g["p2"], angle=g["angle"], text=g["text"],
                               dimstyle="EZDXF", override=ov, dxfattribs={"layer": e["layer"]}).render()
        elif e["kind"] == "dim_diam":
            msp.add_diameter_dim(center=g["center"], radius=g["radius"], angle=g["angle"], text=g["text"],
                                 dimstyle="EZDXF", override=ov, dxfattribs={"layer": e["layer"]}).render()
    doc.saveas(path)
    return path


TEXT_H = 3.5


def _xy(v) -> list:
    return [float(v.x), float(v.y)]


def read_dxf(path: str) -> dict:
    """Flat summary for tests / the future ingestion stage: raw entities plus native dimension info."""
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    out = {"insunits": doc.header.get("$INSUNITS", 0), "lines": [], "circles": [], "arcs": [], "texts": [], "dims": []}
    for e in msp:
        t, lay, lt = e.dxftype(), e.dxf.layer, e.dxf.get("linetype", "BYLAYER")
        if t == "LINE":
            out["lines"].append({"p": _xy(e.dxf.start) + _xy(e.dxf.end), "layer": lay, "ltype": lt})
        elif t == "CIRCLE":
            out["circles"].append({"c": _xy(e.dxf.center), "r": e.dxf.radius, "layer": lay, "ltype": lt})
        elif t == "ARC":
            out["arcs"].append({"c": _xy(e.dxf.center), "r": e.dxf.radius, "a0": e.dxf.start_angle,
                                "a1": e.dxf.end_angle, "layer": lay, "ltype": lt})
        elif t in ("TEXT", "MTEXT"):
            out["texts"].append({"text": e.dxf.text if t == "TEXT" else e.text, "at": _xy(e.dxf.insert), "layer": lay})
        elif t == "DIMENSION":
            out["dims"].append({"measurement": float(e.get_measurement()), "text": e.dxf.get("text", "<>"),
                                "layer": lay, "type": e.dimtype})
    return out
