import cadquery as cq
import math
import numpy as np

def build_wheel_cad():
    # Outer tire
    tire = cq.Workplane("XY").circle(68.0).extrude(45.0)
    tire_inner = cq.Workplane("XY").circle(48.0).extrude(47.0).translate((0, 0, -1.0))
    tire = tire.cut(tire_inner)

    # Rim barrel
    rim = cq.Workplane("XY").circle(48.0).extrude(40.0)
    rim_bore = cq.Workplane("XY").circle(42.0).extrude(42.0).translate((0, 0, -1.0))
    rim = rim.cut(rim_bore)

    # Front wheel disk (where spokes and hub live) at z=28 to 38
    disk = cq.Workplane("XY").workplane(offset=26.0).circle(44.0).extrude(10.0)

    # Center hub bore
    center_bore = cq.Workplane("XY").workplane(offset=25.0).circle(7.0).extrude(13.0)
    disk = disk.cut(center_bore)

    # 5 Lug bolt holes
    for i in range(5):
        ang = math.radians(90.0 + i * 72.0)
        bx, by = 25.0 * math.cos(ang), 25.0 * math.sin(ang)
        hole = cq.Workplane("XY").workplane(offset=25.0).circle(3.0).extrude(13.0).translate((bx, by, 0))
        counterbore = cq.Workplane("XY").workplane(offset=32.0).circle(5.0).extrude(5.0).translate((bx, by, 0))
        disk = disk.cut(hole).cut(counterbore)

    # 5 Spoke cutout windows
    for i in range(5):
        a_center = math.radians(90.0 + 36.0 + i * 72.0)
        r_inner, r_outer = 17.0, 39.0
        arc_inner_pts = [(r_inner * math.cos(a_center + math.radians(step)),
                          r_inner * math.sin(a_center + math.radians(step)))
                         for step in np.linspace(-15, 15, 6)]
        arc_outer_pts = [(r_outer * math.cos(a_center + math.radians(step)),
                          r_outer * math.sin(a_center + math.radians(step)))
                         for step in np.linspace(17, -17, 8)]
        poly_pts = arc_inner_pts + arc_outer_pts
        window = cq.Workplane("XY").workplane(offset=25.0).polyline(poly_pts).close().extrude(13.0)
        disk = disk.cut(window)

    wheel = tire.union(rim).union(disk)
    return wheel

def build_butterfly_cad():
    # Body: abdomen + thorax + head
    ab = cq.Workplane("XY").polyline([(0, 0), (4, 15), (3.5, 40), (0, 48)]).close().revolve(360, (0, 0, 0), (0, 1, 0))
    th = cq.Workplane("XY").polyline([(0, 46), (5.5, 54), (0, 62)]).close().revolve(360, (0, 0, 0), (0, 1, 0))
    head = cq.Workplane("XY").polyline([(0, 61), (4, 65), (0, 69)]).close().revolve(360, (0, 0, 0), (0, 1, 0))
    body = ab.union(th).union(head)

    # Wings (left and right with slight dihedral)
    fw_pts = [
        (4, 52), (18, 56), (42, 68), (75, 86), (92, 94),
        (90, 84), (86, 74), (80, 64), (72, 54), (60, 46),
        (46, 40), (26, 38), (4, 44)
    ]
    wing_l = cq.Workplane("XY").workplane(offset=2.0).polyline(fw_pts).close().extrude(2.0).rotate((0, 50, 0), (0, 50, 1), 6.0)
    fw_pts_r = [(-p[0], p[1]) for p in fw_pts]
    wing_r = cq.Workplane("XY").workplane(offset=2.0).polyline(fw_pts_r).close().extrude(2.0).rotate((0, 50, 0), (0, 50, 1), -6.0)

    hw_pts = [
        (4, 42), (24, 38), (48, 32), (68, 22), (72, 12),
        (64, -2), (48, -12), (32, -14), (18, -6), (4, 15)
    ]
    hwing_l = cq.Workplane("XY").workplane(offset=1.5).polyline(hw_pts).close().extrude(1.8).rotate((0, 20, 0), (0, 20, 1), 4.0)
    hw_pts_r = [(-p[0], p[1]) for p in hw_pts]
    hwing_r = cq.Workplane("XY").workplane(offset=1.5).polyline(hw_pts_r).close().extrude(1.8).rotate((0, 20, 0), (0, 20, 1), -4.0)

    butterfly = body.union(wing_l).union(wing_r).union(hwing_l).union(hwing_r)
    return butterfly

def build_cat_cad():
    # Cat profile extrusion with rounding
    cat_outline = [
        (10, 0), (8, 3), (10, 6), (14, 7), (18, 7), (21, 6), (24, 7), (28, 6), (30, 2), (28, 0),
        (26, 17), (24, 37), (20, 57), (18, 71),
        (21, 83), (18, 87), (14, 91), (16, 95), (20, 96), (25, 96),
        (29, 102), (34, 108), (31, 130),
        (40, 121), (46, 111),
        (50, 113), (56, 132),
        (62, 119), (65, 107),
        (68, 95), (74, 81), (82, 67), (92, 57), (104, 50), (117, 43), (128, 33),
        (136, 19), (134, 7), (130, 0),
        (118, 0), (112, 7), (104, 10), (92, 10), (80, 13), (70, 20),
        (60, 20), (50, 17), (42, 11), (36, 0)
    ]
    cat_body = cq.Workplane("XY").polyline(cat_outline).close().extrude(32.0)

    # Curved tail
    tail_pts = [
        (128, 13), (140, 7), (152, 5), (164, 9), (170, 19), (168, 33), (160, 47), (150, 59), (142, 65),
        (139, 63), (146, 56), (154, 43), (160, 31), (160, 19), (154, 13), (144, 12), (132, 17), (126, 17)
    ]
    tail = cq.Workplane("XY").workplane(offset=10.0).polyline(tail_pts).close().extrude(12.0)
    cat = cat_body.union(tail)
    return cat

def build_engine_cad():
    # Crankcase & Block
    block = cq.Workplane("XY").box(120.0, 70.0, 110.0, centered=(True, True, False)).translate((0, 0, 20.0))
    bore = cq.Workplane("XY").circle(42.0).extrude(120.0).translate((0, 0, 15.0))
    block = block.cut(bore)

    # Piston inside bore
    piston = cq.Workplane("XY").circle(41.5).extrude(30.0).translate((0, 0, 75.0))
    piston_pin_hole = cq.Workplane("YZ").circle(9.0).extrude(90.0).translate((-45.0, 0, 88.0))
    piston = piston.cut(piston_pin_hole)

    # Connecting rod
    rod = cq.Workplane("XZ").rect(14.0, 60.0).extrude(16.0).translate((0, -8.0, 55.0))

    # Crankshaft journal
    crank = cq.Workplane("YZ").circle(18.0).extrude(80.0).translate((-40.0, 0, 25.0))

    # Cylinder head
    head = cq.Workplane("XY").box(130.0, 75.0, 45.0, centered=(True, True, False)).translate((0, 0, 130.0))
    # Two valve guides
    v1 = cq.Workplane("XY").circle(14.0).extrude(25.0).translate((-20.0, 0, 130.0))
    v2 = cq.Workplane("XY").circle(14.0).extrude(25.0).translate((20.0, 0, 130.0))

    engine = block.union(piston).union(rod).union(crank).union(head).union(v1).union(v2)
    return engine

def build_f35_cad():
    # Chined fuselage
    fuse = cq.Workplane("XY").polyline([
        (0, 85), (5, 75), (10, 60), (16, 45), (24, 30), (22, 10), (18, -10),
        (15, -45), (8, -55), (0, -58),
        (-8, -55), (-15, -45), (-18, -10), (-22, 10), (-24, 30), (-16, 45), (-10, 60), (-5, 75)
    ]).close().extrude(18.0)

    # Cockpit canopy bubble
    canopy = cq.Workplane("XY").workplane(offset=15.0).ellipse(6.0, 18.0).extrude(9.0).translate((0, 42.0, 0))

    # Delta Wings
    wing_pts = [(18, 15), (53.5, -15), (53.5, -25), (20, -38), (15, -42)]
    wing_l = cq.Workplane("XY").workplane(offset=6.0).polyline(wing_pts).close().extrude(3.5)
    wing_pts_r = [(-p[0], p[1]) for p in wing_pts]
    wing_r = cq.Workplane("XY").workplane(offset=6.0).polyline(wing_pts_r).close().extrude(3.5)

    # Twin canted vertical stabilizers
    tail_l = cq.Workplane("XY").workplane(offset=12.0).polyline([(14, -30), (24, -58), (22, -62), (12, -45)]).close().extrude(28.0).rotate((0, 0, 0), (0, 1, 0), 20.0)
    tail_r = cq.Workplane("XY").workplane(offset=12.0).polyline([(-14, -30), (-24, -58), (-22, -62), (-12, -45)]).close().extrude(28.0).rotate((0, 0, 0), (0, 1, 0), -20.0)

    # Horizontal tailerons
    htail_l = cq.Workplane("XY").workplane(offset=8.0).polyline([(14, -45), (35, -65), (33, -72), (12, -62)]).close().extrude(2.5)
    htail_r = cq.Workplane("XY").workplane(offset=8.0).polyline([(-14, -45), (-35, -65), (-33, -72), (-12, -62)]).close().extrude(2.5)

    # Engine exhaust nozzle
    nozzle = cq.Workplane("XZ").circle(8.0).extrude(15.0).translate((0, -65.0, 9.0))

    jet = fuse.union(canopy).union(wing_l).union(wing_r).union(tail_l).union(tail_r).union(htail_l).union(htail_r).union(nozzle)
    return jet

models = [
    ("Wheel", build_wheel_cad),
    ("Butterfly", build_butterfly_cad),
    ("Cat", build_cat_cad),
    ("Engine", build_engine_cad),
    ("F-35 Jet", build_f35_cad),
]

for name, builder in models:
    obj = builder()
    val = obj.val()
    print(f"{name:12}: Valid={val.isValid()}  Watertight={val.Solids()[0].Volume() > 0}  Volume={val.Volume():.1f} mm3")
