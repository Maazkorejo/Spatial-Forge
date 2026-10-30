"""Convert a drawing_gen prismatic spec (gen_parts.py) into a FeatureTree. Used to prove the executor reproduces
independently generated ground-truth solids; turned parts need a revolve base (not supported yet)."""
from model import (At, Block, Chamfer, FeatureTree, Hole, PocketRect, Ref, Slot)


def _mn(x):
    return Ref(ref="min", offset=float(x))


def from_spec(spec: dict) -> FeatureTree:
    if spec["family"] != "prismatic":
        raise ValueError("only prismatic parts are supported (turned parts need a revolve base)")
    W, D, H = spec["W"], spec["D"], spec["H"]
    feats = []
    for f in spec["features"]:
        t = f["type"]
        if t == "notch":
            feats.append(PocketRect(face="top", at=At(u=Ref(ref="max", offset=-f["nx"] / 2), v=Ref(ref="max", offset=-f["ny"] / 2)),
                                    size_u=f["nx"], size_v=f["ny"], depth="through"))
        elif t == "step":
            feats.append(PocketRect(face="top", at=At(u=_mn(f["sx"] / 2), v=Ref(ref="center")), size_u=f["sx"], size_v="full", depth=f["sz"]))
        elif t == "chamfer":
            feats.append(Chamfer(selector="top_edges", distance=f["c"]))
        elif t == "hole_z":
            feats.append(Hole(face="top", at=At(u=_mn(f["x"]), v=_mn(f["y"])), d=f["d"]))
        elif t == "slot_z":
            feats.append(Slot(face="top", at=At(u=_mn(f["x"]), v=_mn(f["y"])), length=f["length"], width=f["width"]))
        elif t == "counterbore":                               # a blind hole of the counterbore diameter
            feats.append(Hole(face="top", at=At(u=_mn(f["x"]), v=_mn(f["y"])), d=f["d"], depth=f["depth"]))
        elif t in ("pocket_top", "pocket_bottom"):
            feats.append(PocketRect(face="top" if t == "pocket_top" else "bottom",
                                    at=At(u=_mn(f["x"] + f["sx"] / 2), v=_mn(f["y"] + f["sy"] / 2)),
                                    size_u=f["sx"], size_v=f["sy"], depth=f["depth"]))
        elif t == "hole_y":
            feats.append(Hole(face="front", at=At(u=_mn(f["x"]), v=_mn(f["z"])), d=f["d"]))
        else:
            raise ValueError(f"unknown feature type {t}")
    return FeatureTree(base=Block(W=W, D=D, H=H), features=feats)
