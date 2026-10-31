"""
Definitions of 50 mechanical CAD parts across Tools, Objects, and Difficult CADs.
Each function returns (doc, title, part_no, (bx, by, bw, bh))
"""
import math
import numpy as np
import ezdxf

def make_01_open_wrench_17mm():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_arc, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 17mm Open End Wrench
    # Head at (30, 50), 15 degree angled jaw
    head_cx, head_cy = 30.0, 50.0
    r_head = 22.0
    jaw_w = 17.0
    jaw_d = 20.0
    ang = math.radians(15.0)
    ca, sa = math.cos(ang), math.sin(ang)

    # Handle
    handle_pts = [
        (46.0, 56.0), (145.0, 54.0),
        (155.0, 50.0), (145.0, 46.0),
        (46.0, 44.0)
    ]
    # Jaw notch
    p_throat = (head_cx - 4.0 * ca, head_cy - 4.0 * sa)
    j1 = (p_throat[0] - (jaw_w / 2.0) * -sa, p_throat[1] - (jaw_w / 2.0) * ca)
    j2 = (p_throat[0] + (jaw_w / 2.0) * -sa, p_throat[1] + (jaw_w / 2.0) * ca)
    j1_out = (j1[0] - jaw_d * ca, j1[1] - jaw_d * sa)
    j2_out = (j2[0] - jaw_d * ca, j2[1] - jaw_d * sa)

    profile = [
        j1_out, j1, j2, j2_out,
        (head_cx + r_head * math.cos(math.radians(105)), head_cy + r_head * math.sin(math.radians(105))),
        (46.0, 56.0), (145.0, 54.0),
        (158.0, 50.0), (145.0, 46.0),
        (46.0, 44.0),
        (head_cx + r_head * math.cos(math.radians(255)), head_cy + r_head * math.sin(math.radians(255))),
    ]
    add_lines(msp, profile, closed=True)
    # Hang hole
    add_circle(msp, 150.0, 50.0, 4.0)
    # Side view below
    msp.add_line((20.0, 15.0), (158.0, 15.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((158.0, 15.0), (158.0, 21.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((158.0, 21.0), (20.0, 21.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((20.0, 21.0), (20.0, 15.0), dxfattribs={"layer": "VISIBLE"})
    # Dimensions
    add_dim(msp, (10.0, 50.0), (158.0, 50.0), 30.0, "150 mm")
    add_dim(msp, (20.0, 15.0), (20.0, 21.0), -12.0, "6 mm", vertical=True)
    return doc, "Open-End Wrench 17mm", "TOOL-001", (10.0, 10.0, 160.0, 75.0)

def make_02_combination_spanner():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Left: Open Jaw (width 14mm), Right: 12-point box ring (OD 26mm, ID 15mm)
    # Open jaw
    p1 = (18.0, 60.0); p2 = (30.0, 55.0); p3 = (30.0, 45.0); p4 = (18.0, 40.0)
    p5 = (34.0, 38.0); p6 = (130.0, 46.0); p7 = (130.0, 54.0); p8 = (34.0, 62.0)
    add_lines(msp, [p1, p2, p3, p4, p5, (130.0, 46.0), (145.0, 40.0), (160.0, 50.0), (145.0, 60.0), (130.0, 54.0), p8], closed=True)
    # Box ring internal star (12 points)
    box_cx, box_cy = 150.0, 50.0
    add_circle(msp, box_cx, box_cy, 12.0, centerlines=True)
    star_pts = []
    for i in range(12):
        a1 = math.radians(i * 30.0)
        a2 = math.radians(i * 30.0 + 15.0)
        star_pts.append((box_cx + 8.5 * math.cos(a1), box_cy + 8.5 * math.sin(a1)))
        star_pts.append((box_cx + 7.0 * math.cos(a2), box_cy + 7.0 * math.sin(a2)))
    add_lines(msp, star_pts, closed=True)
    # Side view
    msp.add_line((18.0, 18.0), (162.0, 18.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((162.0, 18.0), (162.0, 24.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((162.0, 24.0), (18.0, 24.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((18.0, 24.0), (18.0, 18.0), dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (18.0, 50.0), (162.0, 50.0), 28.0, "160 mm")
    return doc, "Combination Spanner 14mm", "TOOL-002", (10.0, 10.0, 165.0, 75.0)

def make_03_adjustable_crescent_wrench():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Fixed jaw body + worm gear cavity (16x10mm)
    body = [
        (20.0, 65.0), (20.0, 52.0), (32.0, 48.0), (36.0, 36.0), (45.0, 38.0),
        (52.0, 46.0), (160.0, 44.0), (170.0, 50.0), (160.0, 54.0), (52.0, 54.0),
        (48.0, 68.0), (32.0, 68.0)
    ]
    add_lines(msp, body, closed=True)
    # Worm gear cutout
    add_rect(msp, 36.0, 48.0, 14.0, 8.0, layer="VISIBLE")
    # Movable jaw guide lines (hidden)
    msp.add_line((20.0, 52.0), (38.0, 52.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((20.0, 60.0), (38.0, 60.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    # Hang hole
    add_circle(msp, 162.0, 49.0, 4.5)
    # Side view
    add_rect(msp, 20.0, 15.0, 150.0, 8.0)
    add_dim(msp, (20.0, 50.0), (170.0, 50.0), 30.0, "175 mm")
    return doc, "Adjustable Crescent Wrench 8in", "TOOL-003", (10.0, 10.0, 175.0, 75.0)

def make_04_hex_allen_key_l_wrench():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 8mm Hex key with 90 deg bend: Long leg 110mm, short leg 35mm
    add_lines(msp, [
        (30.0, 80.0), (38.0, 80.0), (38.0, 42.0),
        (65.0, 42.0), (65.0, 34.0), (30.0, 34.0)
    ], closed=True)
    # End view: Hexagon 8mm across flats
    hcx, hcy = 100.0, 55.0
    hr = 4.62
    hex_pts = [(hcx + hr * math.cos(math.radians(i * 60)), hcy + hr * math.sin(math.radians(i * 60))) for i in range(6)]
    add_lines(msp, hex_pts, closed=True)
    msp.add_text("HEX 8mm", height=3.0, dxfattribs={"layer": "VISIBLE", "insert": (90.0, 42.0)})
    add_dim(msp, (30.0, 80.0), (30.0, 34.0), -12.0, "110 mm", vertical=True)
    add_dim(msp, (30.0, 34.0), (65.0, 34.0), -12.0, "35 mm")
    return doc, "Hex Allen Key 8mm L-Wrench", "TOOL-004", (10.0, 15.0, 120.0, 85.0)

def make_05_machinist_precision_square():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Heavy base beam 90x25x15mm, thin blade 140x25x3mm
    beam = [(30.0, 30.0), (120.0, 30.0), (120.0, 55.0), (55.0, 55.0), (55.0, 150.0), (30.0, 150.0)]
    add_lines(msp, beam, closed=True)
    # Corner relief notch (dia 3mm at inner corner (55, 55))
    add_circle(msp, 55.0, 55.0, 2.0, layer="VISIBLE", centerlines=False)
    # Pin fasteners
    add_circle(msp, 42.5, 42.5, 2.5)
    add_circle(msp, 42.5, 75.0, 2.5)
    # Side view
    add_rect(msp, 140.0, 30.0, 16.0, 120.0)
    add_dim(msp, (30.0, 30.0), (120.0, 30.0), -14.0, "90 mm")
    add_dim(msp, (30.0, 30.0), (30.0, 150.0), -14.0, "120 mm", vertical=True)
    return doc, "Machinist Precision Try Square", "TOOL-005", (10.0, 10.0, 165.0, 160.0)

def make_06_vernier_caliper():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Main beam with fixed external jaw and internal knife jaw
    beam = [
        (20.0, 60.0), (180.0, 60.0), (180.0, 44.0),
        (55.0, 44.0), (52.0, 15.0), (36.0, 15.0), (40.0, 44.0),
        (20.0, 44.0), (20.0, 75.0), (34.0, 75.0), (36.0, 60.0)
    ]
    add_lines(msp, beam, closed=True)
    # Sliding carriage outline
    add_rect(msp, 75.0, 40.0, 38.0, 24.0, layer="HIDDEN", ltype="DASHED")
    # Movable jaw
    add_lines(msp, [(75.0, 40.0), (75.0, 15.0), (90.0, 15.0), (92.0, 40.0)], closed=False, layer="HIDDEN", ltype="DASHED")
    # Scale graduations
    for x in range(65, 175, 5):
        h = 4.0 if x % 10 == 0 else 2.5
        msp.add_line((x, 60.0), (x, 60.0 - h), dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (20.0, 60.0), (180.0, 60.0), 22.0, "150 mm Range")
    return doc, "Vernier Caliper 150mm", "TOOL-006", (10.0, 10.0, 190.0, 85.0)

def make_07_slotted_flathead_screwdriver():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Tip (20, 50), Shank (28 to 110), Handle (110 to 180)
    pts = [
        (20.0, 52.0), (20.0, 48.0), (28.0, 47.0), (110.0, 47.0),
        (112.0, 43.0), (120.0, 40.0), (165.0, 38.0), (178.0, 45.0),
        (180.0, 50.0), (178.0, 55.0), (165.0, 62.0), (120.0, 60.0),
        (112.0, 57.0), (110.0, 53.0), (28.0, 53.0)
    ]
    add_lines(msp, pts, closed=True)
    # Handle flutes
    for yf in (45.0, 50.0, 55.0):
        msp.add_line((125.0, yf), (160.0, yf), dxfattribs={"layer": "VISIBLE"})
    add_circle(msp, 172.0, 50.0, 3.0)
    add_dim(msp, (20.0, 50.0), (180.0, 50.0), 22.0, "180 mm")
    return doc, "Slotted Flathead Screwdriver", "TOOL-007", (10.0, 20.0, 185.0, 65.0)

def make_08_ball_peen_hammer_head():
    from generate_50_test_cads import create_cad_doc, add_lines, add_arc, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Left face (flat striking), Center eye hole, Right hemisphere ball peen
    body = [
        (30.0, 62.0), (45.0, 58.0), (75.0, 58.0), (95.0, 56.0),
        (95.0, 44.0), (75.0, 42.0), (45.0, 42.0), (30.0, 38.0)
    ]
    add_lines(msp, body, closed=False)
    msp.add_line((30.0, 38.0), (30.0, 62.0), dxfattribs={"layer": "VISIBLE"})
    # Hemispherical peen at right
    add_arc(msp, 95.0, 50.0, 6.0, 270.0, 90.0)
    # Oval eye hole in center
    add_lines(msp, [(55.0, 55.0), (65.0, 55.0), (68.0, 50.0), (65.0, 45.0), (55.0, 45.0), (52.0, 50.0)], closed=True)
    # Side view: round striking face dia 26mm
    add_circle(msp, 130.0, 50.0, 13.0)
    add_dim(msp, (30.0, 50.0), (101.0, 50.0), 22.0, "100 mm")
    return doc, "Ball-Peen Hammer Head", "TOOL-008", (15.0, 20.0, 140.0, 65.0)

def make_09_claw_hammer_head():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Striking face at left, curved claw split at right
    pts = [
        (30.0, 60.0), (45.0, 56.0), (75.0, 56.0), (95.0, 75.0),
        (105.0, 80.0), (98.0, 72.0), (88.0, 50.0), (98.0, 28.0),
        (105.0, 20.0), (95.0, 25.0), (75.0, 44.0), (45.0, 44.0), (30.0, 40.0)
    ]
    add_lines(msp, pts, closed=True)
    # Eye hole
    add_circle(msp, 60.0, 50.0, 7.0)
    add_dim(msp, (30.0, 50.0), (105.0, 50.0), 40.0, "115 mm")
    return doc, "Curved Claw Hammer Head", "TOOL-009", (15.0, 10.0, 115.0, 90.0)

def make_10_needle_nose_pliers():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Tapered nose, pivot boss, dual handles
    upper = [
        (25.0, 51.5), (70.0, 55.0), (78.0, 62.0), (88.0, 58.0),
        (160.0, 72.0), (170.0, 70.0), (150.0, 56.0), (85.0, 46.0), (25.0, 48.5)
    ]
    add_lines(msp, upper, closed=True)
    add_circle(msp, 80.0, 50.0, 9.0)
    add_circle(msp, 80.0, 50.0, 4.0)
    add_dim(msp, (25.0, 50.0), (170.0, 50.0), 30.0, "160 mm")
    return doc, "Needle-Nose Pliers", "TOOL-010", (15.0, 25.0, 170.0, 70.0)

def make_11_slip_joint_utility_pliers():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    body = [
        (25.0, 58.0), (45.0, 58.0), (55.0, 52.0), (70.0, 65.0),
        (155.0, 74.0), (165.0, 70.0), (145.0, 55.0), (80.0, 44.0),
        (55.0, 42.0), (45.0, 46.0), (25.0, 46.0)
    ]
    add_lines(msp, body, closed=True)
    # Figure 8 peanut slot
    add_circle(msp, 72.0, 54.0, 4.0)
    add_circle(msp, 78.0, 54.0, 4.0)
    add_dim(msp, (25.0, 50.0), (165.0, 50.0), 32.0, "165 mm")
    return doc, "Slip-Joint Utility Pliers", "TOOL-011", (15.0, 25.0, 165.0, 70.0)

def make_12_precision_forceps_tweezers():
    from generate_50_test_cads import create_cad_doc, add_lines, add_arc, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    top_blade = [(20.0, 50.8), (70.0, 54.0), (140.0, 54.0), (148.0, 50.0)]
    bot_blade = [(20.0, 49.2), (70.0, 46.0), (140.0, 46.0), (148.0, 50.0)]
    add_lines(msp, top_blade, closed=False)
    add_lines(msp, bot_blade, closed=False)
    # Grip pads
    for x in range(80, 120, 4):
        msp.add_line((x, 52.0), (x, 54.0), dxfattribs={"layer": "VISIBLE"})
        msp.add_line((x, 46.0), (x, 48.0), dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (20.0, 50.0), (148.0, 50.0), 16.0, "130 mm")
    return doc, "Precision Forceps Tweezers", "TOOL-012", (10.0, 30.0, 150.0, 45.0)

def make_13_chisel_scraper_blade():
    from generate_50_test_cads import create_cad_doc, add_lines, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 25mm wide flat scraper blade with 25 deg bevel, shoulder, and tang
    pts = [
        (30.0, 62.5), (30.0, 37.5), (85.0, 37.5), (85.0, 45.0),
        (135.0, 46.0), (140.0, 50.0), (135.0, 54.0), (85.0, 55.0), (85.0, 62.5)
    ]
    add_lines(msp, pts, closed=True)
    # Bevel line
    msp.add_line((34.0, 62.5), (34.0, 37.5), dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (30.0, 50.0), (140.0, 50.0), 22.0, "110 mm")
    add_dim(msp, (30.0, 37.5), (30.0, 62.5), -14.0, "25 mm", vertical=True)
    return doc, "Woodworking Chisel Blade", "TOOL-013", (15.0, 20.0, 140.0, 60.0)

def make_14_trapezoid_utility_knife_blade():
    from generate_50_test_cads import create_cad_doc, add_lines, add_arc, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 60x19mm trapezoid blade with dual notches
    pts = [
        (30.0, 35.0), (90.0, 35.0), (76.0, 54.0),
        (66.0, 54.0), (64.0, 51.0), (60.0, 51.0), (58.0, 54.0),
        (48.0, 54.0), (46.0, 51.0), (42.0, 51.0), (40.0, 54.0),
        (44.0, 54.0)
    ]
    add_lines(msp, [(30.0, 35.0), (90.0, 35.0), (76.0, 54.0), (66.0, 54.0), (64.0, 50.0), (58.0, 50.0), (56.0, 54.0), (48.0, 54.0), (46.0, 50.0), (40.0, 50.0), (38.0, 54.0), (30.0, 54.0)], closed=True)
    # Razor ground edge line
    msp.add_line((30.0, 38.0), (90.0, 38.0), dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (30.0, 35.0), (90.0, 35.0), -12.0, "60 mm")
    add_dim(msp, (30.0, 35.0), (30.0, 54.0), -12.0, "19 mm", vertical=True)
    return doc, "Trapezoid Utility Knife Blade", "TOOL-014", (15.0, 15.0, 95.0, 60.0)

def make_15_tap_wrench_handle():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Center block 30x24mm, left arm 65x10mm, right arm 65x10mm
    add_rect(msp, 65.0, 38.0, 30.0, 24.0)
    add_rect(msp, 15.0, 44.0, 50.0, 12.0)
    add_rect(msp, 95.0, 44.0, 50.0, 12.0)
    # V-clamp square jaw in center
    add_lines(msp, [(74.0, 50.0), (80.0, 56.0), (86.0, 50.0), (80.0, 44.0)], closed=True)
    add_dim(msp, (15.0, 50.0), (145.0, 50.0), 24.0, "130 mm")
    return doc, "Tap Wrench Handle", "TOOL-015", (10.0, 20.0, 150.0, 60.0)


# ==========================================
# GROUP 2: MECHANICAL OBJECTS (16 to 35)
# ==========================================

def make_16_spur_gear_16t():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    # 16-tooth involute gear
    n = 16
    r_pitch = 40.0
    r_tip = 45.0
    r_root = 34.0
    gear_pts = []
    for i in range(n):
        ang = i * (2.0 * math.pi / n)
        d_ang = (2.0 * math.pi / n)
        gear_pts.append((cx + r_root * math.cos(ang), cy + r_root * math.sin(ang)))
        gear_pts.append((cx + r_tip * math.cos(ang + d_ang * 0.25), cy + r_tip * math.sin(ang + d_ang * 0.25)))
        gear_pts.append((cx + r_tip * math.cos(ang + d_ang * 0.50), cy + r_tip * math.sin(ang + d_ang * 0.50)))
        gear_pts.append((cx + r_root * math.cos(ang + d_ang * 0.75), cy + r_root * math.sin(ang + d_ang * 0.75)))
    add_lines(msp, gear_pts, closed=True)
    # Center bore dia 20mm + 6mm keyway
    add_circle(msp, cx, cy, 10.0)
    add_rect(msp, cx - 3.0, cy + 8.0, 6.0, 4.0)
    # 4 lightening holes dia 12mm
    for a in (45, 135, 225, 315):
        rad = math.radians(a)
        add_circle(msp, cx + 24.0 * math.cos(rad), cy + 24.0 * math.sin(rad), 5.5)
    # Side view at (130, 60)
    add_rect(msp, 125.0, 15.0, 20.0, 90.0)
    add_rect(msp, 120.0, 45.0, 30.0, 30.0) # Hub
    # Centerline bore in side view
    msp.add_line((120.0, 50.0), (150.0, 50.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((120.0, 70.0), (150.0, 70.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    add_dim(msp, (cx - 45.0, cy), (cx + 45.0, cy), 55.0, "OD 90 mm")
    return doc, "Spur Gear 16T Involute", "OBJ-016", (10.0, 10.0, 160.0, 115.0)

def make_17_roller_chain_sprocket():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    n = 14
    r_pitch = 38.0
    r_tip = 43.0
    r_root = 32.0
    pts = []
    for i in range(n):
        ang = i * (2.0 * math.pi / n)
        da = (2.0 * math.pi / n)
        pts.append((cx + r_root * math.cos(ang), cy + r_root * math.sin(ang)))
        pts.append((cx + r_tip * math.cos(ang + da * 0.35), cy + r_tip * math.sin(ang + da * 0.35)))
        pts.append((cx + r_root * math.cos(ang + da * 0.70), cy + r_root * math.sin(ang + da * 0.70)))
    add_lines(msp, pts, closed=True)
    add_circle(msp, cx, cy, 12.0)
    add_dim(msp, (cx - r_tip, cy), (cx + r_tip, cy), 50.0, "OD 86 mm")
    return doc, "Roller Chain Sprocket 14T", "OBJ-017", (10.0, 10.0, 110.0, 110.0)

def make_18_ansi_pipe_flange_6bolt():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # OD 120mm, Flow Bore dia 50mm, PCD 95mm with 6 bolts dia 14mm
    cx, cy = 60.0, 60.0
    add_circle(msp, cx, cy, 60.0) # Outer flange
    add_circle(msp, cx, cy, 25.0) # Central bore
    add_circle(msp, cx, cy, 47.5, layer="CENTER", centerlines=False) # Pitch Circle
    for i in range(6):
        ang = math.radians(i * 60)
        bx = cx + 47.5 * math.cos(ang)
        by = cy + 47.5 * math.sin(ang)
        add_circle(msp, bx, by, 7.0)
    # Side cross-section view
    add_rect(msp, 140.0, 0.0, 18.0, 120.0)
    add_rect(msp, 158.0, 15.0, 6.0, 90.0) # Raised face
    # Center bore cutout
    msp.add_line((140.0, 35.0), (164.0, 35.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((140.0, 85.0), (164.0, 85.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    add_dim(msp, (cx, cy - 60.0), (cx, cy + 60.0), -20.0, "OD 120 mm", vertical=True)
    return doc, "ANSI 6-Bolt Pipe Flange", "OBJ-018", (0.0, -10.0, 180.0, 145.0)

def make_19_rhombus_flanged_bearing_mount():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 70.0, 50.0
    # Rhombus body: center circle R32mm, ear tabs extending to x=20 and x=120
    pts = [
        (20.0, 50.0), (45.0, 75.0), (95.0, 75.0),
        (120.0, 50.0), (95.0, 25.0), (45.0, 25.0)
    ]
    add_lines(msp, pts, closed=True)
    add_circle(msp, cx, cy, 24.0) # Spherical bearing seat
    add_circle(msp, cx, cy, 18.0) # Shaft clearance
    # 2 bolt holes on ears
    add_circle(msp, 28.0, 50.0, 5.5)
    add_circle(msp, 112.0, 50.0, 5.5)
    add_dim(msp, (28.0, 50.0), (112.0, 50.0), 32.0, "84 mm Bolt Centers")
    return doc, "Rhombus 2-Bolt Flange Bearing", "OBJ-019", (10.0, 15.0, 125.0, 75.0)

def make_20_engine_piston_head():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Piston profile 75x70mm, 3 ring lands, pin bore dia 20mm
    add_rect(msp, 25.0, 25.0, 75.0, 70.0)
    # Ring grooves on sides
    for yr in (82.0, 86.0, 90.0):
        msp.add_line((25.0, yr), (29.0, yr), dxfattribs={"layer": "VISIBLE"})
        msp.add_line((96.0, yr), (100.0, yr), dxfattribs={"layer": "VISIBLE"})
    # Gudgeon pin bore
    add_circle(msp, 62.5, 55.0, 10.0)
    # Skirt cutout
    add_rect(msp, 40.0, 25.0, 45.0, 12.0)
    add_dim(msp, (25.0, 25.0), (100.0, 25.0), -12.0, "75 mm Bore")
    add_dim(msp, (100.0, 25.0), (100.0, 95.0), 12.0, "70 mm Height", vertical=True)
    return doc, "Engine Piston Head", "OBJ-020", (15.0, 10.0, 105.0, 95.0)

def make_21_h_beam_connecting_rod():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Big end (crank) at (35, 50), Small end (wristpin) at (145, 50)
    # Big end outer dia 56mm, bore 40mm
    add_circle(msp, 35.0, 50.0, 28.0)
    add_circle(msp, 35.0, 50.0, 20.0)
    # Small end outer dia 34mm, bore 20mm
    add_circle(msp, 145.0, 50.0, 17.0)
    add_circle(msp, 145.0, 50.0, 10.0)
    # H-beam shank
    add_lines(msp, [(58.0, 57.0), (130.0, 55.0), (130.0, 45.0), (58.0, 43.0)], closed=True)
    add_lines(msp, [(62.0, 54.0), (126.0, 52.0), (126.0, 48.0), (62.0, 46.0)], closed=True) # H recess
    add_dim(msp, (35.0, 50.0), (145.0, 50.0), 36.0, "110 mm C-to-C")
    return doc, "H-Beam Connecting Rod", "OBJ-021", (5.0, 15.0, 165.0, 80.0)

def make_22_stepped_transmission_shaft():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Shaft steps along X: (0 to 30 dia 18), (30 to 70 dia 28), (70 to 110 dia 36), (110 to 150 dia 28), (150 to 180 dia 18)
    cy = 50.0
    steps = [
        (20.0, cy - 9.0, 30.0, 18.0),
        (50.0, cy - 14.0, 40.0, 28.0),
        (90.0, cy - 18.0, 40.0, 36.0),
        (130.0, cy - 14.0, 40.0, 28.0),
        (170.0, cy - 9.0, 30.0, 18.0),
    ]
    for x, y, w, h in steps:
        add_rect(msp, x, y, w, h)
    # Centerline
    msp.add_line((10.0, cy), (210.0, cy), dxfattribs={"layer": "CENTER", "linetype": "CENTER"})
    # Keyseat
    add_rect(msp, 100.0, cy + 12.0, 20.0, 4.0)
    add_dim(msp, (20.0, cy), (200.0, cy), 30.0, "180 mm Total")
    return doc, "Stepped Transmission Shaft", "OBJ-022", (10.0, 20.0, 200.0, 65.0)

def make_23_crankshaft_web_counterweight():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Main journal bore at (50, 40) dia 30mm, Crankpin at (50, 85) dia 24mm
    # Swept crescent counterweight below y=40
    web_pts = [
        (30.0, 85.0), (30.0, 40.0), (10.0, 15.0), (50.0, 5.0),
        (90.0, 15.0), (70.0, 40.0), (70.0, 85.0)
    ]
    add_lines(msp, web_pts, closed=True)
    add_circle(msp, 50.0, 40.0, 15.0) # Main bore
    add_circle(msp, 50.0, 85.0, 12.0) # Pin bore
    add_dim(msp, (50.0, 40.0), (50.0, 85.0), -25.0, "45 mm Stroke", vertical=True)
    return doc, "Crankshaft Web Counterweight", "OBJ-023", (5.0, 0.0, 100.0, 100.0)

def make_24_centrifugal_pump_impeller():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_arc, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    r_shroud = 50.0
    r_hub = 16.0
    add_circle(msp, cx, cy, r_shroud)
    add_circle(msp, cx, cy, r_hub)
    add_circle(msp, cx, cy, 9.0) # Bore
    # 6 backward curved vanes
    for i in range(6):
        a = i * 60.0
        rad = math.radians(a)
        p_in = (cx + r_hub * math.cos(rad), cy + r_hub * math.sin(rad))
        p_out = (cx + r_shroud * math.cos(rad + 0.8), cy + r_shroud * math.sin(rad + 0.8))
        msp.add_line(p_in, p_out, dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (cx - r_shroud, cy), (cx + r_shroud, cy), 60.0, "OD 100 mm")
    return doc, "Centrifugal Pump Impeller", "OBJ-024", (5.0, 5.0, 120.0, 120.0)

def make_25_airfoil_propeller_blade_hub():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 90.0, 50.0
    # Hub in center dia 24mm
    add_circle(msp, cx, cy, 12.0)
    add_circle(msp, cx, cy, 4.0)
    # Blade left
    blade_l = [(cx - 12.0, cy + 6.0), (cx - 70.0, cy + 14.0), (cx - 78.0, cy), (cx - 70.0, cy - 8.0), (cx - 12.0, cy - 4.0)]
    # Blade right
    blade_r = [(cx + 12.0, cy + 4.0), (cx + 70.0, cy + 8.0), (cx + 78.0, cy), (cx + 70.0, cy - 14.0), (cx + 12.0, cy - 6.0)]
    add_lines(msp, blade_l, closed=True)
    add_lines(msp, blade_r, closed=True)
    add_dim(msp, (cx - 78.0, cy), (cx + 78.0, cy), 25.0, "156 mm Propeller Span")
    return doc, "UAV Propeller Hub", "OBJ-025", (5.0, 20.0, 175.0, 65.0)

def make_26_butterfly_valve_throttle_disc():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    r = 45.0
    add_circle(msp, cx, cy, r)
    # Shaft lug across middle
    add_rect(msp, cx - 12.0, cy - 48.0, 24.0, 96.0)
    add_circle(msp, cx, cy - 35.0, 6.0)
    add_circle(msp, cx, cy + 35.0, 6.0)
    add_dim(msp, (cx - r, cy), (cx + r, cy), 54.0, "DN90 Valve Disc")
    return doc, "Butterfly Valve Throttle Disc", "OBJ-026", (10.0, 10.0, 110.0, 110.0)

def make_27_pillow_block_bearing_housing():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Cast iron pedestal base 130x20mm, center housing dome R32mm, center bore dia 35mm
    base = [
        (15.0, 20.0), (145.0, 20.0), (145.0, 36.0), (115.0, 38.0),
        (112.0, 75.0), (48.0, 75.0), (45.0, 38.0), (15.0, 36.0)
    ]
    add_lines(msp, base, closed=True)
    # Center bore
    add_circle(msp, 80.0, 52.0, 18.0)
    add_circle(msp, 80.0, 52.0, 12.0)
    # 2 slotted bolt holes on base
    add_rect(msp, 22.0, 24.0, 14.0, 8.0)
    add_rect(msp, 124.0, 24.0, 14.0, 8.0)
    add_dim(msp, (15.0, 20.0), (145.0, 20.0), -14.0, "130 mm Base")
    return doc, "Pillow Block Bearing Housing", "OBJ-027", (10.0, 5.0, 145.0, 85.0)

def make_28_heavy_duty_gusseted_l_bracket():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 90x90x14mm L-bracket with gusset triangle
    bracket = [
        (25.0, 25.0), (115.0, 25.0), (115.0, 39.0),
        (39.0, 39.0), (39.0, 115.0), (25.0, 115.0)
    ]
    add_lines(msp, bracket, closed=True)
    # Gusset rib line
    msp.add_line((39.0, 95.0), (95.0, 39.0), dxfattribs={"layer": "VISIBLE"})
    # Holes
    add_circle(msp, 32.0, 75.0, 4.5)
    add_circle(msp, 32.0, 100.0, 4.5)
    add_circle(msp, 75.0, 32.0, 4.5)
    add_circle(msp, 100.0, 32.0, 4.5)
    add_dim(msp, (25.0, 25.0), (115.0, 25.0), -12.0, "90 mm")
    add_dim(msp, (25.0, 25.0), (25.0, 115.0), -12.0, "90 mm", vertical=True)
    return doc, "Gusseted Heavy-Duty L-Bracket", "OBJ-028", (10.0, 10.0, 115.0, 115.0)

def make_29_u_clevis_shackle_yoke():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # U-shape clevis: width 70mm, height 85mm, throat 30mm
    pts = [
        (30.0, 25.0), (90.0, 25.0), (90.0, 95.0), (70.0, 95.0),
        (70.0, 55.0), (50.0, 55.0), (50.0, 95.0), (30.0, 95.0)
    ]
    add_lines(msp, pts, closed=True)
    # Pin holes through ears
    add_circle(msp, 40.0, 80.0, 6.0)
    add_circle(msp, 80.0, 80.0, 6.0)
    # Threaded eye on base
    add_circle(msp, 60.0, 35.0, 5.0)
    add_dim(msp, (30.0, 25.0), (90.0, 25.0), -12.0, "60 mm Width")
    add_dim(msp, (90.0, 25.0), (90.0, 95.0), 12.0, "70 mm Throat", vertical=True)
    return doc, "U-Clevis Rigging Shackle", "OBJ-029", (15.0, 10.0, 95.0, 95.0)

def make_30_deep_throat_c_clamp_frame():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # C-frame with throat depth 75mm, span 110mm
    pts = [
        (25.0, 25.0), (105.0, 25.0), (105.0, 42.0), (55.0, 42.0),
        (55.0, 100.0), (105.0, 100.0), (105.0, 125.0), (25.0, 125.0)
    ]
    add_lines(msp, pts, closed=True)
    # Threaded screw collar at top
    add_rect(msp, 95.0, 125.0, 18.0, 10.0)
    # Anvil pad at bottom
    add_rect(msp, 95.0, 34.0, 18.0, 8.0)
    add_dim(msp, (25.0, 25.0), (25.0, 125.0), -14.0, "100 mm Span", vertical=True)
    return doc, "Deep-Throat C-Clamp Frame", "OBJ-030", (10.0, 15.0, 120.0, 130.0)

def make_31_geneva_wheel_4slot():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_arc, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    r_outer = 45.0
    slot_w = 8.0
    slot_d = 25.0
    # 4 radial slots at 0, 90, 180, 270 deg
    add_circle(msp, cx, cy, 10.0) # Hub bore
    # Outline with 4 concave locking arcs
    for i in range(4):
        a = i * 90.0
        rad = math.radians(a)
        ca, sa = math.cos(rad), math.sin(rad)
        # slot
        p1 = (cx + (r_outer - slot_d) * ca - 4.0 * -sa, cy + (r_outer - slot_d) * sa - 4.0 * ca)
        p2 = (cx + r_outer * ca - 4.0 * -sa, cy + r_outer * sa - 4.0 * ca)
        p3 = (cx + r_outer * ca + 4.0 * -sa, cy + r_outer * sa + 4.0 * ca)
        p4 = (cx + (r_outer - slot_d) * ca + 4.0 * -sa, cy + (r_outer - slot_d) * sa + 4.0 * ca)
        msp.add_line(p1, p2, dxfattribs={"layer": "VISIBLE"})
        msp.add_line(p3, p4, dxfattribs={"layer": "VISIBLE"})
        msp.add_line(p1, p4, dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (cx - r_outer, cy), (cx + r_outer, cy), 54.0, "OD 90 mm")
    return doc, "Geneva Indexing Star Wheel", "OBJ-031", (10.0, 10.0, 110.0, 110.0)

def make_32_eccentric_camshaft_lobe():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 50.0
    # Base circle R20mm, nose crest at (60, 85) R10mm
    r_base = 20.0
    r_nose = 9.0
    nose_cy = 78.0
    pts = [
        (cx - r_base, cy), (cx - 16.0, 68.0), (cx - r_nose, nose_cy),
        (cx + r_nose, nose_cy), (cx + 16.0, 68.0), (cx + r_base, cy),
        (cx + 14.0, cy - 14.0), (cx, cy - r_base), (cx - 14.0, cy - 14.0)
    ]
    add_lines(msp, pts, closed=True)
    add_circle(msp, cx, cy, 12.0) # Camshaft center bore
    add_dim(msp, (cx, cy - r_base), (cx, nose_cy + r_nose), 35.0, "57 mm Total Lift", vertical=True)
    return doc, "Eccentric Camshaft Lobe", "OBJ-032", (20.0, 15.0, 85.0, 95.0)

def make_33_v_belt_sheave_pulley():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Front round view + Side V-groove profile
    cx, cy = 55.0, 55.0
    add_circle(msp, cx, cy, 45.0)
    add_circle(msp, cx, cy, 12.0)
    # Side view: V-groove sheave
    v_pts = [
        (120.0, 10.0), (145.0, 10.0), (145.0, 48.0), (138.0, 55.0),
        (145.0, 62.0), (145.0, 100.0), (120.0, 100.0), (120.0, 62.0),
        (127.0, 55.0), (120.0, 48.0)
    ]
    add_lines(msp, v_pts, closed=True)
    add_dim(msp, (cx - 45.0, cy), (cx + 45.0, cy), 55.0, "OD 90 mm")
    return doc, "V-Belt Single Sheave Pulley", "OBJ-033", (5.0, 5.0, 150.0, 115.0)

def make_34_flexible_jaw_coupling_spider():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    r_out = 36.0
    add_circle(msp, cx, cy, r_out)
    add_circle(msp, cx, cy, 11.0) # Shaft bore
    # 3 dog teeth at 120 deg
    for i in range(3):
        a = i * 120.0
        rad = math.radians(a)
        p1 = (cx + 16.0 * math.cos(rad - 0.25), cy + 16.0 * math.sin(rad - 0.25))
        p2 = (cx + 34.0 * math.cos(rad - 0.25), cy + 34.0 * math.sin(rad - 0.25))
        p3 = (cx + 34.0 * math.cos(rad + 0.25), cy + 34.0 * math.sin(rad + 0.25))
        p4 = (cx + 16.0 * math.cos(rad + 0.25), cy + 16.0 * math.sin(rad + 0.25))
        add_lines(msp, [p1, p2, p3, p4], closed=True)
    add_dim(msp, (cx - r_out, cy), (cx + r_out, cy), 45.0, "OD 72 mm")
    return doc, "Flexible Jaw Coupling Half", "OBJ-034", (10.0, 10.0, 105.0, 105.0)

def make_35_splined_hub_collar():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 60.0
    add_circle(msp, cx, cy, 42.0) # Outer collar
    # 8 internal splines
    r_maj = 22.0
    r_min = 17.0
    spline_pts = []
    for i in range(8):
        a = i * (2.0 * math.pi / 8.0)
        da = (2.0 * math.pi / 8.0)
        spline_pts.append((cx + r_min * math.cos(a), cy + r_min * math.sin(a)))
        spline_pts.append((cx + r_maj * math.cos(a + da * 0.25), cy + r_maj * math.sin(a + da * 0.25)))
        spline_pts.append((cx + r_maj * math.cos(a + da * 0.50), cy + r_maj * math.sin(a + da * 0.50)))
        spline_pts.append((cx + r_min * math.cos(a + da * 0.75), cy + r_min * math.sin(a + da * 0.75)))
    add_lines(msp, spline_pts, closed=True)
    add_dim(msp, (cx - 42.0, cy), (cx + 42.0, cy), 50.0, "8-Spline Hub")
    return doc, "Involute Splined Drive Collar", "OBJ-035", (10.0, 10.0, 110.0, 110.0)

# ==========================================
# GROUP 3: DIFFICULT & COMPLEX CADs (36 to 50)
# ==========================================

def make_36_aerospace_wing_spar_rib():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Cambered NACA airfoil profile (chord 180mm) with 3 flanged lightening holes & spar cutouts
    pts = [
        (20.0, 50.0), (40.0, 68.0), (80.0, 75.0), (120.0, 70.0),
        (160.0, 58.0), (195.0, 51.0), (160.0, 46.0), (120.0, 42.0),
        (80.0, 40.0), (40.0, 42.0)
    ]
    add_lines(msp, pts, closed=True)
    # Front & Rear spar cutouts
    msp.add_line((50.0, 71.0), (50.0, 58.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((50.0, 58.0), (58.0, 58.0), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((58.0, 58.0), (58.0, 73.0), dxfattribs={"layer": "VISIBLE"})
    # 3 flanged lightening holes
    add_circle(msp, 85.0, 56.0, 11.0)
    add_circle(msp, 85.0, 56.0, 9.0, layer="HIDDEN", centerlines=False)
    add_circle(msp, 125.0, 55.0, 9.0)
    add_circle(msp, 155.0, 53.0, 6.0)
    add_dim(msp, (20.0, 50.0), (195.0, 50.0), 32.0, "175 mm Chord")
    return doc, "Aerospace Wing Spar Rib", "DIF-036", (10.0, 25.0, 195.0, 65.0)

def make_37_turbine_cascade_guide_vane():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Complex nozzle guide vane with inner/outer shrouds and fir-tree root
    # Vane curved profile
    pts = [
        (30.0, 25.0), (80.0, 25.0), (80.0, 35.0), (62.0, 35.0),
        (75.0, 75.0), (68.0, 115.0), (80.0, 115.0), (80.0, 125.0),
        (30.0, 125.0), (30.0, 115.0), (48.0, 115.0), (52.0, 75.0),
        (42.0, 35.0), (30.0, 35.0)
    ]
    add_lines(msp, pts, closed=True)
    # Fir tree serrations at root (below y=25)
    fir_pts = [
        (35.0, 25.0), (30.0, 20.0), (35.0, 15.0), (30.0, 10.0),
        (75.0, 10.0), (70.0, 15.0), (75.0, 20.0), (70.0, 25.0)
    ]
    add_lines(msp, fir_pts, closed=True)
    add_dim(msp, (30.0, 10.0), (30.0, 125.0), -16.0, "115 mm Height", vertical=True)
    return doc, "Turbine Cascade Guide Vane", "DIF-037", (10.0, 0.0, 95.0, 135.0)

def make_38_f35_stealth_fighter_airframe():
    from generate_50_test_cads import create_cad_doc, add_lines, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Detailed F-35 planform
    f35_planform = [
        (80.0, 135.0), (86.0, 110.0), (96.0, 90.0), (145.0, 52.0),
        (140.0, 42.0), (105.0, 48.0), (110.0, 25.0), (102.0, 18.0),
        (85.0, 26.0), (80.0, 20.0), (75.0, 26.0), (58.0, 18.0),
        (50.0, 25.0), (55.0, 48.0), (20.0, 42.0), (15.0, 52.0),
        (64.0, 90.0), (74.0, 110.0)
    ]
    add_lines(msp, f35_planform, closed=True)
    # Canopy and chine lines
    add_lines(msp, [(77.0, 120.0), (83.0, 120.0), (84.0, 95.0), (76.0, 95.0)], closed=True)
    msp.add_line((80.0, 15.0), (80.0, 140.0), dxfattribs={"layer": "CENTER", "linetype": "CENTER"})
    add_dim(msp, (15.0, 52.0), (145.0, 52.0), -38.0, "Wingspan 130 mm")
    add_dim(msp, (80.0, 20.0), (80.0, 135.0), 75.0, "Length 115 mm", vertical=True)
    return doc, "F-35 Stealth Fighter Airframe", "DIF-038", (5.0, 5.0, 155.0, 140.0)

def make_39_high_density_heat_sink_comb():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # Base 140x12mm with 16 tall cooling fins (height 45mm, thickness 2mm)
    base_x, base_y = 20.0, 20.0
    add_rect(msp, base_x, base_y, 140.0, 12.0)
    for i in range(16):
        fx = base_x + 4.0 + i * 8.5
        add_rect(msp, fx, base_y + 12.0, 2.2, 45.0)
    add_dim(msp, (base_x, base_y), (base_x + 140.0, base_y), -12.0, "140 mm Base")
    add_dim(msp, (base_x, base_y), (base_x, base_y + 57.0), -12.0, "57 mm Fins", vertical=True)
    return doc, "High-Density Heat Sink Array", "DIF-039", (10.0, 5.0, 155.0, 75.0)

def make_40_planetary_gear_carrier_cage():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 65.0, 65.0
    # Triangular 3-pinion carrier with rounded lobes
    pts = []
    for i in range(3):
        a = i * 120.0 - 90.0
        rad = math.radians(a)
        # lobe center at radius 38mm
        lx = cx + 38.0 * math.cos(rad)
        ly = cy + 38.0 * math.sin(rad)
        for sub_a in range(-60, 61, 30):
            s_rad = math.radians(a + sub_a)
            pts.append((lx + 16.0 * math.cos(s_rad), ly + 16.0 * math.sin(s_rad)))
    add_lines(msp, pts, closed=True)
    add_circle(msp, cx, cy, 14.0) # Sun shaft bore
    # 3 planet pin bores
    for i in range(3):
        a = math.radians(i * 120.0 - 90.0)
        add_circle(msp, cx + 38.0 * math.cos(a), cy + 38.0 * math.sin(a), 7.0)
    add_dim(msp, (cx - 50.0, cy), (cx + 50.0, cy), 65.0, "115 mm Carrier PCD")
    return doc, "Planetary Gear Carrier Cage", "DIF-040", (5.0, 5.0, 120.0, 130.0)

def make_41_orthographic_3rd_angle_test_block():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # ANSI 3-view layout: Front (25, 25), Top (25, 85), Right Side (95, 25)
    # Front: step ramp
    add_lines(msp, [(25.0, 25.0), (75.0, 25.0), (75.0, 45.0), (50.0, 65.0), (25.0, 65.0)], closed=True)
    # Front hidden bore
    msp.add_line((35.0, 25.0), (35.0, 48.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((45.0, 25.0), (45.0, 48.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    # Top view:
    add_rect(msp, 25.0, 80.0, 50.0, 35.0)
    msp.add_line((50.0, 80.0), (50.0, 115.0), dxfattribs={"layer": "VISIBLE"}) # Ramp ridge
    add_circle(msp, 40.0, 97.5, 5.0) # Top bore
    # Right view:
    add_rect(msp, 90.0, 25.0, 35.0, 40.0)
    msp.add_line((90.0, 45.0), (125.0, 45.0), dxfattribs={"layer": "VISIBLE"})
    add_dim(msp, (25.0, 25.0), (75.0, 25.0), -12.0, "50 mm Width")
    add_dim(msp, (25.0, 25.0), (25.0, 65.0), -12.0, "40 mm Height", vertical=True)
    return doc, "Orthographic 3rd-Angle Block", "DIF-041", (10.0, 10.0, 125.0, 115.0)

def make_42_iso_1st_angle_complex_bracket():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # ISO 1st angle layout: Front at (30, 80), Top BELOW front at (30, 20), Right to LEFT at (105, 80)
    # Front view: L-bracket with corner chamfer and counterbore
    add_lines(msp, [(30.0, 80.0), (80.0, 80.0), (80.0, 95.0), (48.0, 95.0), (48.0, 125.0), (30.0, 125.0)], closed=True)
    # Top view below
    add_rect(msp, 30.0, 25.0, 50.0, 35.0)
    msp.add_line((48.0, 25.0), (48.0, 60.0), dxfattribs={"layer": "VISIBLE"})
    add_circle(msp, 65.0, 42.5, 4.5)
    # Side view
    add_rect(msp, 100.0, 80.0, 35.0, 45.0)
    add_dim(msp, (30.0, 80.0), (80.0, 80.0), 55.0, "50 mm ISO Front")
    return doc, "ISO 1st-Angle Multi-View Bracket", "DIF-042", (15.0, 10.0, 130.0, 125.0)

def make_43_hydraulic_multiport_manifold_block():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 120x90mm block with P, T, A, B ports, O-ring counterbores, and cross galleys
    add_rect(msp, 20.0, 20.0, 120.0, 90.0)
    ports = [
        ("P", 50.0, 45.0), ("T", 50.0, 85.0),
        ("A", 110.0, 45.0), ("B", 110.0, 85.0)
    ]
    for label, px, py in ports:
        add_circle(msp, px, py, 11.0) # O-ring recess
        add_circle(msp, px, py, 6.0)  # Port hole
        msp.add_text(label, height=4.0, dxfattribs={"layer": "VISIBLE", "insert": (px - 2.0, py - 2.0)})
    # Internal cross-drilling galleys (hidden lines)
    msp.add_line((20.0, 45.0), (140.0, 45.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((20.0, 85.0), (140.0, 85.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    # 4 mounting bolt holes
    for mx, my in ((28.0, 28.0), (132.0, 28.0), (132.0, 102.0), (28.0, 102.0)):
        add_circle(msp, mx, my, 4.0)
    add_dim(msp, (20.0, 20.0), (140.0, 20.0), -12.0, "120 mm Manifold")
    return doc, "Hydraulic Multi-Port Manifold", "DIF-043", (10.0, 5.0, 140.0, 115.0)

def make_44_stepped_concentric_turned_cylinder():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cy = 60.0
    # Multi-step turned cylinder with interior counterbore and cross bleed port
    steps = [
        (20.0, cy - 12.0, 25.0, 24.0),
        (45.0, cy - 20.0, 35.0, 40.0),
        (80.0, cy - 28.0, 50.0, 56.0),
        (130.0, cy - 16.0, 30.0, 32.0),
    ]
    for x, y, w, h in steps:
        add_rect(msp, x, y, w, h)
    # Centerline
    msp.add_line((10.0, cy), (170.0, cy), dxfattribs={"layer": "CENTER", "linetype": "CENTER"})
    # Internal blind bore (hidden lines)
    msp.add_line((20.0, cy - 6.0), (105.0, cy - 6.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((20.0, cy + 6.0), (105.0, cy + 6.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    msp.add_line((105.0, cy - 6.0), (105.0, cy + 6.0), dxfattribs={"layer": "HIDDEN", "linetype": "DASHED"})
    add_dim(msp, (20.0, cy), (160.0, cy), 38.0, "140 mm Turned Cylinder")
    return doc, "Stepped Concentric Turned Cylinder", "DIF-044", (10.0, 25.0, 160.0, 80.0)

def make_45_geneva_drive_crank_arm():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_arc, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 60.0, 50.0
    # Crank hub at (60, 50) dia 30mm, drive pin at (105, 50) dia 8mm, locking crescent disc R24mm
    add_circle(msp, cx, cy, 15.0) # Hub
    add_circle(msp, cx, cy, 8.0)  # Shaft bore
    # Drive arm extending to pin
    arm_pts = [(cx, cy + 12.0), (105.0, cy + 8.0), (112.0, cy), (105.0, cy - 8.0), (cx, cy - 12.0)]
    add_lines(msp, arm_pts, closed=True)
    add_circle(msp, 105.0, 50.0, 4.0) # Drive roller pin
    # Locking dwell disc arc
    add_arc(msp, cx, cy, 22.0, 45.0, 315.0)
    add_dim(msp, (cx, cy), (105.0, cy), -25.0, "45 mm Crank Radius")
    return doc, "Geneva Drive Crank Arm", "DIF-045", (15.0, 15.0, 115.0, 75.0)

def make_46_ventilated_disc_brake_rotor():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 65.0, 65.0
    r_disc = 55.0
    r_hat = 30.0
    add_circle(msp, cx, cy, r_disc) # Outer friction rotor
    add_circle(msp, cx, cy, r_disc - 16.0) # Friction inner track
    add_circle(msp, cx, cy, r_hat) # Mounting hat
    add_circle(msp, cx, cy, 15.0) # Hub register
    # 5-stud lug bolt circle
    for i in range(5):
        a = math.radians(i * 72.0 - 90.0)
        add_circle(msp, cx + 22.0 * math.cos(a), cy + 22.0 * math.sin(a), 4.5)
    # Cooling vent drillings in friction track
    for i in range(12):
        a = math.radians(i * 30.0)
        add_circle(msp, cx + 47.0 * math.cos(a), cy + 47.0 * math.sin(a), 2.2, centerlines=False)
    add_dim(msp, (cx - r_disc, cy), (cx + r_disc, cy), 65.0, "OD 110 mm Rotor")
    return doc, "Ventilated Disc Brake Rotor", "DIF-046", (5.0, 5.0, 125.0, 130.0)

def make_47_automotive_10spoke_alloy_wheel():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 65.0, 65.0
    r_rim = 58.0
    r_bead = 54.0
    add_circle(msp, cx, cy, r_rim)
    add_circle(msp, cx, cy, r_bead)
    add_circle(msp, cx, cy, 18.0) # Hub center
    add_circle(msp, cx, cy, 10.0) # Center bore
    # 5 lug bolt holes
    for i in range(5):
        a = math.radians(i * 72.0 - 90.0)
        add_circle(msp, cx + 14.0 * math.cos(a), cy + 14.0 * math.sin(a), 3.0, centerlines=False)
    # 10 spokes (5 split pairs)
    for i in range(10):
        a = math.radians(i * 36.0)
        p1 = (cx + 18.0 * math.cos(a - 0.08), cy + 18.0 * math.sin(a - 0.08))
        p2 = (cx + 52.0 * math.cos(a - 0.05), cy + 52.0 * math.sin(a - 0.05))
        p3 = (cx + 52.0 * math.cos(a + 0.05), cy + 52.0 * math.sin(a + 0.05))
        p4 = (cx + 18.0 * math.cos(a + 0.08), cy + 18.0 * math.sin(a + 0.08))
        add_lines(msp, [p1, p2, p3, p4], closed=True)
    add_dim(msp, (cx - r_rim, cy), (cx + r_rim, cy), 68.0, "10-Spoke Wheel Face")
    return doc, "Automotive 10-Spoke Alloy Wheel", "DIF-047", (5.0, 5.0, 125.0, 135.0)

def make_48_ribbed_electronics_enclosure_chassis():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 140x95mm avionics enclosure with gasket groove and 6 PCB standoffs
    add_rect(msp, 20.0, 20.0, 140.0, 95.0) # Outer wall
    add_rect(msp, 24.0, 24.0, 132.0, 87.0, layer="VISIBLE") # Tongue groove
    add_rect(msp, 28.0, 28.0, 124.0, 79.0) # Inner cavity
    # 6 internal PCB threaded standoffs
    standoffs = [
        (40.0, 40.0), (90.0, 40.0), (140.0, 40.0),
        (40.0, 95.0), (90.0, 95.0), (140.0, 95.0)
    ]
    for sx, sy in standoffs:
        add_circle(msp, sx, sy, 4.0)
        add_circle(msp, sx, sy, 1.8, centerlines=False)
    add_dim(msp, (20.0, 20.0), (160.0, 20.0), -12.0, "140 mm Chassis")
    add_dim(msp, (20.0, 20.0), (20.0, 115.0), -12.0, "95 mm Width", vertical=True)
    return doc, "Ribbed Electronics Chassis", "DIF-048", (10.0, 5.0, 160.0, 120.0)

def make_49_swashplate_control_ring():
    from generate_50_test_cads import create_cad_doc, add_lines, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    cx, cy = 65.0, 65.0
    r_ring = 38.0
    add_circle(msp, cx, cy, r_ring)
    add_circle(msp, cx, cy, 26.0) # Uniball bearing seat
    # 3 control horns extending outward at 120 deg
    for i in range(3):
        a = math.radians(i * 120.0)
        ca, sa = math.cos(a), math.sin(a)
        horn_pts = [
            (cx + r_ring * ca - 7.0 * -sa, cy + r_ring * sa - 7.0 * ca),
            (cx + 56.0 * ca - 5.0 * -sa, cy + 56.0 * sa - 5.0 * ca),
            (cx + 62.0 * ca, cy + 62.0 * sa),
            (cx + 56.0 * ca + 5.0 * -sa, cy + 56.0 * sa + 5.0 * ca),
            (cx + r_ring * ca + 7.0 * -sa, cy + r_ring * sa + 7.0 * ca),
        ]
        add_lines(msp, horn_pts, closed=True)
        add_circle(msp, cx + 55.0 * ca, cy + 55.0 * sa, 3.5, centerlines=False)
    add_dim(msp, (cx - 56.0, cy), (cx + 56.0, cy), 70.0, "Swashplate Horn Ring")
    return doc, "Helicopter Swashplate Control Ring", "DIF-049", (5.0, 5.0, 130.0, 140.0)

def make_50_multicylinder_engine_head_gasket():
    from generate_50_test_cads import create_cad_doc, add_lines, add_rect, add_circle, add_dim
    doc = create_cad_doc()
    msp = doc.modelspace()
    # 4-cylinder engine head gasket (180x70mm)
    add_rect(msp, 20.0, 20.0, 175.0, 70.0)
    # 4 combustion chambers dia 32mm
    for i in range(4):
        bx = 45.0 + i * 38.0
        by = 55.0
        add_circle(msp, bx, by, 16.0) # Combustion fire ring
        add_circle(msp, bx, by, 14.5, layer="HIDDEN", centerlines=False)
        # Coolant holes around each bore
        add_circle(msp, bx, by + 23.0, 3.0, centerlines=False)
        add_circle(msp, bx, by - 23.0, 3.0, centerlines=False)
    # 10 main head bolt holes
    for i in range(5):
        hx = 28.0 + i * 38.0
        add_circle(msp, hx, 28.0, 4.5)
        add_circle(msp, hx, 82.0, 4.5)
    add_dim(msp, (20.0, 20.0), (195.0, 20.0), -14.0, "175 mm 4-Cylinder Gasket")
    return doc, "4-Cylinder Engine Head Gasket", "DIF-050", (10.0, 5.0, 195.0, 95.0)

# Master registry of all 50 builders
ALL_BUILDERS = [
    # Tools (1 to 15)
    ("01_tools", "01_open_wrench_17mm", make_01_open_wrench_17mm),
    ("01_tools", "02_combination_spanner", make_02_combination_spanner),
    ("01_tools", "03_adjustable_crescent_wrench", make_03_adjustable_crescent_wrench),
    ("01_tools", "04_hex_allen_key_l_wrench", make_04_hex_allen_key_l_wrench),
    ("01_tools", "05_machinist_precision_square", make_05_machinist_precision_square),
    ("01_tools", "06_vernier_caliper", make_06_vernier_caliper),
    ("01_tools", "07_slotted_flathead_screwdriver", make_07_slotted_flathead_screwdriver),
    ("01_tools", "08_ball_peen_hammer_head", make_08_ball_peen_hammer_head),
    ("01_tools", "09_claw_hammer_head", make_09_claw_hammer_head),
    ("01_tools", "10_needle_nose_pliers", make_10_needle_nose_pliers),
    ("01_tools", "11_slip_joint_utility_pliers", make_11_slip_joint_utility_pliers),
    ("01_tools", "12_precision_forceps_tweezers", make_12_precision_forceps_tweezers),
    ("01_tools", "13_chisel_scraper_blade", make_13_chisel_scraper_blade),
    ("01_tools", "14_trapezoid_utility_knife_blade", make_14_trapezoid_utility_knife_blade),
    ("01_tools", "15_tap_wrench_handle", make_15_tap_wrench_handle),

    # Objects (16 to 35)
    ("02_objects", "16_spur_gear_16t", make_16_spur_gear_16t),
    ("02_objects", "17_roller_chain_sprocket", make_17_roller_chain_sprocket),
    ("02_objects", "18_ansi_pipe_flange_6bolt", make_18_ansi_pipe_flange_6bolt),
    ("02_objects", "19_rhombus_flanged_bearing_mount", make_19_rhombus_flanged_bearing_mount),
    ("02_objects", "20_engine_piston_head", make_20_engine_piston_head),
    ("02_objects", "21_h_beam_connecting_rod", make_21_h_beam_connecting_rod),
    ("02_objects", "22_stepped_transmission_shaft", make_22_stepped_transmission_shaft),
    ("02_objects", "23_crankshaft_web_counterweight", make_23_crankshaft_web_counterweight),
    ("02_objects", "24_centrifugal_pump_impeller", make_24_centrifugal_pump_impeller),
    ("02_objects", "25_airfoil_propeller_blade_hub", make_25_airfoil_propeller_blade_hub),
    ("02_objects", "26_butterfly_valve_throttle_disc", make_26_butterfly_valve_throttle_disc),
    ("02_objects", "27_pillow_block_bearing_housing", make_27_pillow_block_bearing_housing),
    ("02_objects", "28_heavy_duty_gusseted_l_bracket", make_28_heavy_duty_gusseted_l_bracket),
    ("02_objects", "29_u_clevis_shackle_yoke", make_29_u_clevis_shackle_yoke),
    ("02_objects", "30_deep_throat_c_clamp_frame", make_30_deep_throat_c_clamp_frame),
    ("02_objects", "31_geneva_wheel_4slot", make_31_geneva_wheel_4slot),
    ("02_objects", "32_eccentric_camshaft_lobe", make_32_eccentric_camshaft_lobe),
    ("02_objects", "33_v_belt_sheave_pulley", make_33_v_belt_sheave_pulley),
    ("02_objects", "34_flexible_jaw_coupling_spider", make_34_flexible_jaw_coupling_spider),
    ("02_objects", "35_splined_hub_collar", make_35_splined_hub_collar),

    # Difficult CADs (36 to 50)
    ("03_difficult_cads", "36_aerospace_wing_spar_rib", make_36_aerospace_wing_spar_rib),
    ("03_difficult_cads", "37_turbine_cascade_guide_vane", make_37_turbine_cascade_guide_vane),
    ("03_difficult_cads", "38_f35_stealth_fighter_airframe", make_38_f35_stealth_fighter_airframe),
    ("03_difficult_cads", "39_high_density_heat_sink_comb", make_39_high_density_heat_sink_comb),
    ("03_difficult_cads", "40_planetary_gear_carrier_cage", make_40_planetary_gear_carrier_cage),
    ("03_difficult_cads", "41_orthographic_3rd_angle_test_block", make_41_orthographic_3rd_angle_test_block),
    ("03_difficult_cads", "42_iso_1st_angle_complex_bracket", make_42_iso_1st_angle_complex_bracket),
    ("03_difficult_cads", "43_hydraulic_multiport_manifold_block", make_43_hydraulic_multiport_manifold_block),
    ("03_difficult_cads", "44_stepped_concentric_turned_cylinder", make_44_stepped_concentric_turned_cylinder),
    ("03_difficult_cads", "45_geneva_drive_crank_arm", make_45_geneva_drive_crank_arm),
    ("03_difficult_cads", "46_ventilated_disc_brake_rotor", make_46_ventilated_disc_brake_rotor),
    ("03_difficult_cads", "47_automotive_10spoke_alloy_wheel", make_47_automotive_10spoke_alloy_wheel),
    ("03_difficult_cads", "48_ribbed_electronics_enclosure_chassis", make_48_ribbed_electronics_enclosure_chassis),
    ("03_difficult_cads", "49_swashplate_control_ring", make_49_swashplate_control_ring),
    ("03_difficult_cads", "50_multicylinder_engine_head_gasket", make_50_multicylinder_engine_head_gasket),
]
