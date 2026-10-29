"""
SpatialForge - Multi-Sample DXF and Rendered Picture Generator
Generates CAD-accurate DXF drawings and high-resolution blueprint/drawing pictures for:
1. 06_wheel_technical_drawing (Automotive Forged Alloy Wheel)
2. 07_butterfly_sketch (Monarch Butterfly CNC Vector Contour)
3. 08_cat_silhouette_sketch (Felis Catus Geometric CAD Profile)
4. 09_engine_technical_drawing (2.0L DOHC Internal Combustion Engine Assembly)
5. 10_f35_jet_technical_drawing (Lockheed Martin F-35A Stealth Fighter)
"""

import os
import shutil
import math
import numpy as np
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TEST_DXF_DIR = os.path.join(REPO_ROOT, "test_dxf_drawings")
DESKTOP_DIR = r"C:\Users\maazk\Desktop\DXF_Test_Samples"
os.makedirs(TEST_DXF_DIR, exist_ok=True)
os.makedirs(DESKTOP_DIR, exist_ok=True)

def setup_dxf(title, dwg_no, scale="1:1"):
    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()

    # Create layers
    doc.layers.add("VISIBLE", color=7)
    doc.layers.add("HIDDEN", color=1)
    doc.layers.add("CENTER", color=3)
    doc.layers.add("DIMENSIONS", color=4)
    doc.layers.add("BORDER", color=8)
    doc.layers.add("TEXT", color=2)

    # Standard Sheet Border (280 x 200 mm)
    msp.add_lwpolyline([(0, 0), (280, 0), (280, 200), (0, 200)], close=True, dxfattribs={"layer": "BORDER"})
    msp.add_lwpolyline([(8, 8), (272, 8), (272, 192), (8, 192)], close=True, dxfattribs={"layer": "BORDER"})

    # Title Block
    msp.add_lwpolyline([(165, 8), (272, 8), (272, 38), (165, 38)], close=True, dxfattribs={"layer": "BORDER"})
    msp.add_line((165, 24), (272, 24), dxfattribs={"layer": "BORDER"})
    msp.add_line((220, 8), (220, 24), dxfattribs={"layer": "BORDER"})

    msp.add_text("SPATIAL FORGE CAD SYSTEM", height=2.2, dxfattribs={"layer": "TEXT"}).set_placement((168, 31))
    msp.add_text(title, height=2.6, dxfattribs={"layer": "VISIBLE"}).set_placement((168, 26))
    msp.add_text(f"DWG: {dwg_no}", height=2.0, dxfattribs={"layer": "DIMENSIONS"}).set_placement((168, 18))
    msp.add_text(f"SCALE: {scale}", height=2.0, dxfattribs={"layer": "DIMENSIONS"}).set_placement((168, 12))
    msp.add_text("UNITS: mm", height=2.0, dxfattribs={"layer": "DIMENSIONS"}).set_placement((224, 18))
    msp.add_text("STATUS: RELEASED", height=2.0, dxfattribs={"layer": "TEXT"}).set_placement((224, 12))

    return doc, msp

def render_dxf_to_png(doc, png_path):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=250)
    fig.patch.set_facecolor("#15181e")
    ax.set_facecolor("#15181e")

    msp = doc.modelspace()
    for e in msp:
        layer = e.dxf.layer.upper()
        if layer == "BORDER":
            color, lw, ls = "#4a5568", 1.0, "-"
        elif layer == "VISIBLE":
            color, lw, ls = "#f7fafc", 1.3, "-"
        elif layer == "HIDDEN":
            color, lw, ls = "#f87171", 0.9, "--"
        elif layer == "CENTER":
            color, lw, ls = "#34d399", 0.8, "-."
        elif layer == "DIMENSIONS":
            color, lw, ls = "#38bdf8", 0.8, "-"
        elif layer == "TEXT":
            color, lw, ls = "#fbbf24", 0.9, "-"
        else:
            color, lw, ls = "#cbd5e1", 1.0, "-"

        dxftype = e.dxftype()
        if dxftype == "LINE":
            p1, p2 = e.dxf.start, e.dxf.end
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=lw, linestyle=ls)
        elif dxftype == "CIRCLE":
            c, r = e.dxf.center, e.dxf.radius
            circle = plt.Circle((c.x, c.y), r, color=color, fill=False, linewidth=lw, linestyle=ls)
            ax.add_patch(circle)
        elif dxftype == "ARC":
            c, r = e.dxf.center, e.dxf.radius
            th1, th2 = e.dxf.start_angle, e.dxf.end_angle
            arc = patches.Arc((c.x, c.y), 2*r, 2*r, angle=0, theta1=th1, theta2=th2, color=color, linewidth=lw, linestyle=ls)
            ax.add_patch(arc)
        elif dxftype in ("LWPOLYLINE", "POLYLINE"):
            pts = list(e.get_points())
            if len(pts) > 1:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if e.is_closed:
                    xs.append(pts[0][0])
                    ys.append(pts[0][1])
                ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls)
        elif dxftype == "TEXT":
            p = e.dxf.insert
            txt = e.dxf.text
            h = e.dxf.height
            ax.text(p.x, p.y, txt, color=color, fontsize=max(6, h * 2.8), family="sans-serif", weight="bold")

    ax.set_xlim(-5, 285)
    ax.set_ylim(-5, 205)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(png_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Rendered: {png_path}")

# =============================================================================
# 1. WHEEL (Automotive Forged Alloy Wheel)
# =============================================================================
def build_wheel():
    doc, msp = setup_dxf("FORGED ALLOY WHEEL (18x8.5J ET35)", "SF-WHL-2026-06", "1:2")
    cx, cy = 85.0, 105.0

    msp.add_line((cx - 75, cy), (cx + 75, cy), dxfattribs={"layer": "CENTER"})
    msp.add_line((cx, cy - 75), (cx, cy + 75), dxfattribs={"layer": "CENTER"})

    msp.add_circle((cx, cy), radius=68.0, dxfattribs={"layer": "VISIBLE"})
    for r_tread in (64.0, 60.0, 56.0):
        msp.add_circle((cx, cy), radius=r_tread, dxfattribs={"layer": "HIDDEN"})

    msp.add_circle((cx, cy), radius=48.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, cy), radius=45.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, cy), radius=41.0, dxfattribs={"layer": "VISIBLE"})

    msp.add_circle((cx, cy), radius=16.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, cy), radius=13.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, cy), radius=7.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, cy), radius=4.0, dxfattribs={"layer": "VISIBLE"})

    pcd_r = 25.0
    msp.add_circle((cx, cy), radius=pcd_r, dxfattribs={"layer": "CENTER"})
    for i in range(5):
        ang = math.radians(90.0 + i * 72.0)
        bx = cx + pcd_r * math.cos(ang)
        by = cy + pcd_r * math.sin(ang)
        msp.add_circle((bx, by), radius=2.8, dxfattribs={"layer": "VISIBLE"})
        msp.add_circle((bx, by), radius=4.8, dxfattribs={"layer": "VISIBLE"})

    for i in range(5):
        a_center = math.radians(90.0 + 36.0 + i * 72.0)
        r_inner = 18.5
        r_outer = 39.5

        arc_inner_pts = [(cx + r_inner * math.cos(a_center + math.radians(step)),
                          cy + r_inner * math.sin(a_center + math.radians(step)))
                         for step in np.linspace(-15, 15, 8)]
        arc_outer_pts = [(cx + r_outer * math.cos(a_center + math.radians(step)),
                          cy + r_outer * math.sin(a_center + math.radians(step)))
                         for step in np.linspace(17, -17, 10)]

        poly_pts = arc_inner_pts + arc_outer_pts
        msp.add_lwpolyline(poly_pts, close=True, dxfattribs={"layer": "VISIBLE"})

        spoke_recess = [(cx + (r_inner + 5) * math.cos(a_center + math.radians(step)),
                         cy + (r_inner + 5) * math.sin(a_center + math.radians(step)))
                        for step in np.linspace(-10, 10, 5)] + \
                       [(cx + (r_outer - 4) * math.cos(a_center + math.radians(step)),
                         cy + (r_outer - 4) * math.sin(a_center + math.radians(step)))
                        for step in np.linspace(12, -12, 7)]
        msp.add_lwpolyline(spoke_recess, close=True, dxfattribs={"layer": "HIDDEN"})

    va = math.radians(45.0)
    vx, vy = cx + 43.0 * math.cos(va), cy + 43.0 * math.sin(va)
    msp.add_circle((vx, vy), radius=1.8, dxfattribs={"layer": "VISIBLE"})

    # Side View / Cross-Section (Right)
    sx = 195.0
    msp.add_line((sx - 25, cy), (sx + 45, cy), dxfattribs={"layer": "CENTER"})
    rim_top = [
        (sx - 15, cy + 48), (sx - 12, cy + 45), (sx - 2, cy + 44),
        (sx + 10, cy + 39), (sx + 25, cy + 44), (sx + 32, cy + 48), (sx + 35, cy + 45)
    ]
    msp.add_lwpolyline(rim_top, close=False, dxfattribs={"layer": "VISIBLE"})
    rim_bot = [
        (sx - 15, cy - 48), (sx - 12, cy - 45), (sx - 2, cy - 44),
        (sx + 10, cy - 39), (sx + 25, cy - 44), (sx + 32, cy - 48), (sx + 35, cy - 45)
    ]
    msp.add_lwpolyline(rim_bot, close=False, dxfattribs={"layer": "VISIBLE"})

    tire_top = [(sx - 20, cy + 48), (sx - 24, cy + 58), (sx - 15, cy + 68), (sx + 30, cy + 68), (sx + 40, cy + 58), (sx + 35, cy + 45)]
    msp.add_lwpolyline(tire_top, close=False, dxfattribs={"layer": "VISIBLE"})
    tire_bot = [(sx - 20, cy - 48), (sx - 24, cy - 58), (sx - 15, cy - 68), (sx + 30, cy - 68), (sx + 40, cy - 58), (sx + 35, cy - 45)]
    msp.add_lwpolyline(tire_bot, close=False, dxfattribs={"layer": "VISIBLE"})

    msp.add_line((sx, cy - 25), (sx, cy + 25), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((sx - 8, cy - 18), (sx - 8, cy + 18), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((sx - 8, cy + 18), (sx, cy + 18), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((sx - 8, cy - 18), (sx, cy - 18), dxfattribs={"layer": "VISIBLE"})

    msp.add_text("DIAMETER: 18 INCH [457 mm]", height=2.2, dxfattribs={"layer": "DIMENSIONS"}).set_placement((cx - 30, 25))
    msp.add_text("PCD: 5x114.3 mm", height=2.2, dxfattribs={"layer": "DIMENSIONS"}).set_placement((cx - 18, 18))
    msp.add_text("RIM WIDTH: 8.5J", height=2.2, dxfattribs={"layer": "DIMENSIONS"}).set_placement((sx - 12, 25))
    msp.add_text("OFFSET: ET+35", height=2.2, dxfattribs={"layer": "DIMENSIONS"}).set_placement((sx - 10, 18))

    return doc

# =============================================================================
# 2. BUTTERFLY (Monarch Butterfly CNC Vector Contour)
# =============================================================================
def build_butterfly():
    doc, msp = setup_dxf("MONARCH BUTTERFLY CNC VECTOR CONTOUR", "SF-BF-2026-07", "1:1")
    cx = 140.0

    msp.add_line((cx, 20), (cx, 185), dxfattribs={"layer": "CENTER"})

    msp.add_circle((cx, 126), radius=4.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx - 2.8, 127.5), radius=1.2, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx + 2.8, 127.5), radius=1.2, dxfattribs={"layer": "VISIBLE"})

    ant_l = [(cx - 1.5, 129.5), (cx - 8, 140), (cx - 18, 153), (cx - 28, 162), (cx - 36, 165)]
    ant_r = [(cx + 1.5, 129.5), (cx + 8, 140), (cx + 18, 153), (cx + 28, 162), (cx + 36, 165)]
    msp.add_lwpolyline(ant_l, close=False, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(ant_r, close=False, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx - 36, 165), radius=1.6, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx + 36, 165), radius=1.6, dxfattribs={"layer": "VISIBLE"})

    thorax_pts = [(cx + 5.5 * math.cos(t), 115 + 9.5 * math.sin(t)) for t in np.linspace(0, 2*math.pi, 28)]
    msp.add_lwpolyline(thorax_pts, close=True, dxfattribs={"layer": "VISIBLE"})

    ab_l = [(cx - 4.2 * (1 - 0.45 * (105 - y)/53.0), y) for y in np.linspace(105, 54, 18)] + [(cx, 50)]
    ab_r = [(cx, 50)] + [(cx + 4.2 * (1 - 0.45 * (105 - y)/53.0), y) for y in np.linspace(54, 105, 18)]
    msp.add_lwpolyline(ab_l + ab_r, close=True, dxfattribs={"layer": "VISIBLE"})
    for sy in (98, 91, 84, 77, 70, 63, 57):
        w = 3.8 * (1 - 0.45 * (105 - sy)/53.0)
        msp.add_line((cx - w, sy), (cx + w, sy), dxfattribs={"layer": "VISIBLE"})

    fw_l = [
        (cx - 5, 118), (cx - 15, 126), (cx - 38, 142), (cx - 68, 158), (cx - 96, 166),
        (cx - 94, 155), (cx - 91, 145), (cx - 86, 134), (cx - 80, 124), (cx - 72, 115),
        (cx - 62, 108), (cx - 48, 102), (cx - 28, 98), (cx - 5, 104)
    ]
    fw_r = [(2*cx - p[0], p[1]) for p in fw_l]
    msp.add_lwpolyline(fw_l, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(fw_r, close=True, dxfattribs={"layer": "VISIBLE"})

    def add_fw_venation(sgn):
        dc = [(cx + sgn * 6, 112), (cx + sgn * 24, 116), (cx + sgn * 40, 124), (cx + sgn * 32, 110), (cx + sgn * 18, 106)]
        msp.add_lwpolyline(dc, close=True, dxfattribs={"layer": "VISIBLE"})
        veins = [
            ((cx + sgn * 40, 124), (cx + sgn * 82, 156)),
            ((cx + sgn * 40, 124), (cx + sgn * 88, 142)),
            ((cx + sgn * 36, 117), (cx + sgn * 82, 128)),
            ((cx + sgn * 32, 110), (cx + sgn * 74, 118)),
            ((cx + sgn * 24, 108), (cx + sgn * 60, 108)),
            ((cx + sgn * 18, 106), (cx + sgn * 44, 103))
        ]
        for v0, v1 in veins:
            msp.add_line(v0, v1, dxfattribs={"layer": "VISIBLE"})

    add_fw_venation(-1)
    add_fw_venation(1)

    hw_l = [
        (cx - 5, 102), (cx - 24, 98), (cx - 48, 92), (cx - 68, 82), (cx - 78, 68),
        (cx - 74, 52), (cx - 65, 40), (cx - 52, 32), (cx - 35, 30),
        (cx - 22, 38), (cx - 12, 52), (cx - 4, 68)
    ]
    hw_r = [(2*cx - p[0], p[1]) for p in hw_l]
    msp.add_lwpolyline(hw_l, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(hw_r, close=True, dxfattribs={"layer": "VISIBLE"})

    def add_hw_venation(sgn):
        dc = [(cx + sgn * 5, 94), (cx + sgn * 22, 86), (cx + sgn * 35, 74), (cx + sgn * 22, 65), (cx + sgn * 5, 72)]
        msp.add_lwpolyline(dc, close=True, dxfattribs={"layer": "VISIBLE"})
        veins = [
            ((cx + sgn * 35, 74), (cx + sgn * 72, 68)),
            ((cx + sgn * 32, 70), (cx + sgn * 66, 52)),
            ((cx + sgn * 28, 66), (cx + sgn * 55, 38)),
            ((cx + sgn * 22, 65), (cx + sgn * 38, 32)),
            ((cx + sgn * 14, 68), (cx + sgn * 22, 40))
        ]
        for v0, v1 in veins:
            msp.add_line(v0, v1, dxfattribs={"layer": "VISIBLE"})

    add_hw_venation(-1)
    add_hw_venation(1)

    msp.add_line((cx - 96, 175), (cx + 96, 175), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx - 96, 168), (cx - 96, 178), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx + 96, 168), (cx + 96, 178), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("WINGSPAN: 192.0 mm", height=2.3, dxfattribs={"layer": "DIMENSIONS"}).set_placement((cx - 24, 177))

    msp.add_line((25, 30), (25, 166), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((22, 30), (28, 30), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((22, 166), (28, 166), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("TOTAL HEIGHT: 136.0 mm", height=2.2, dxfattribs={"layer": "DIMENSIONS"}).set_placement((28, 95))

    return doc

# =============================================================================
# 3. CAT (Felis Catus Geometric CAD Profile)
# =============================================================================
def build_cat():
    doc, msp = setup_dxf("FELIS CATUS - CAD VECTOR SILHOUETTE & PROFILE", "SF-CAT-2026-08", "1:1")

    msp.add_line((40, 35), (250, 35), dxfattribs={"layer": "CENTER"})
    msp.add_text("GROUND DATUM (Z=0.0)", height=2.0, dxfattribs={"layer": "TEXT"}).set_placement((45, 30))

    cat_outline = [
        (78, 35), (76, 38), (78, 41), (82, 42), (86, 42), (89, 41), (92, 42), (96, 41), (98, 37), (96, 35),
        (94, 52), (92, 72), (88, 92), (86, 106),
        (89, 118), (86, 122), (82, 126), (84, 130), (88, 131), (93, 131),
        (97, 137), (102, 143), (99, 165),
        (108, 156), (114, 146),
        (118, 148), (124, 167),
        (130, 154), (133, 142),
        (136, 130), (142, 116), (150, 102), (160, 92), (172, 85), (185, 78), (196, 68),
        (204, 54), (202, 42), (198, 35),
        (186, 35), (180, 42), (172, 45), (160, 45), (148, 48), (138, 55),
        (128, 55), (118, 52), (110, 46), (104, 35)
    ]
    msp.add_lwpolyline(cat_outline, close=True, dxfattribs={"layer": "VISIBLE"})

    tail_pts = [
        (196, 48), (208, 42), (220, 40), (232, 44), (238, 54), (236, 68), (228, 82), (218, 94), (210, 100),
        (207, 98), (214, 91), (222, 78), (228, 66), (228, 54), (222, 48), (212, 47), (200, 52), (194, 52)
    ]
    msp.add_lwpolyline(tail_pts, close=True, dxfattribs={"layer": "VISIBLE"})

    msp.add_lwpolyline([(102, 145), (102, 160), (109, 148)], close=False, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline([(120, 148), (124, 162), (128, 146)], close=False, dxfattribs={"layer": "VISIBLE"})

    msp.add_lwpolyline([(95, 132), (99, 135), (104, 133), (100, 130)], close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((99.5, 132.8), radius=1.2, dxfattribs={"layer": "VISIBLE"})

    msp.add_line((86, 125), (62, 128), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((86, 124), (60, 122), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((86, 123), (64, 116), dxfattribs={"layer": "VISIBLE"})

    collar_pts = [(90, 114), (94, 112), (100, 113), (103, 117), (100, 119), (93, 118)]
    msp.add_lwpolyline(collar_pts, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((96.5, 108), radius=2.6, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((96.5, 108), radius=1.2, dxfattribs={"layer": "VISIBLE"})

    msp.add_line((76, 175), (238, 175), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((76, 170), (76, 180), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((238, 170), (238, 180), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("TOTAL LENGTH: 162.0 mm", height=2.3, dxfattribs={"layer": "DIMENSIONS"}).set_placement((135, 177))

    msp.add_line((30, 35), (30, 167), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((26, 35), (34, 35), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((26, 167), (34, 167), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("HEIGHT: 132.0 mm", height=2.3, dxfattribs={"layer": "DIMENSIONS"}).set_placement((33, 100))

    return doc

# =============================================================================
# 4. ENGINE (2.0L DOHC Internal Combustion Engine Cross-Section)
# =============================================================================
def build_engine():
    doc, msp = setup_dxf("2.0L DOHC ENGINE CYLINDER ASSEMBLY", "SF-ENG-2026-09", "1:2")
    cx = 140.0

    msp.add_line((cx, 15), (cx, 190), dxfattribs={"layer": "CENTER"})

    crank_y = 40.0
    msp.add_line((cx - 45, crank_y), (cx + 45, crank_y), dxfattribs={"layer": "CENTER"})
    msp.add_circle((cx, crank_y), radius=16.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, crank_y), radius=8.0, dxfattribs={"layer": "VISIBLE"})

    cw_pts = [
        (cx - 24, crank_y + 12), (cx - 36, crank_y - 8), (cx - 28, crank_y - 24),
        (cx, crank_y - 30), (cx + 28, crank_y - 24), (cx + 36, crank_y - 8), (cx + 24, crank_y + 12)
    ]
    msp.add_lwpolyline(cw_pts, close=False, dxfattribs={"layer": "VISIBLE"})

    c_ang = math.radians(50.0)
    pin_r = 20.0
    px, py = cx + pin_r * math.cos(c_ang), crank_y + pin_r * math.sin(c_ang)
    msp.add_circle((px, py), radius=10.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((px, py), radius=5.5, dxfattribs={"layer": "VISIBLE"})

    pin_wrist_y = 104.0
    msp.add_circle((cx, pin_wrist_y), radius=9.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, pin_wrist_y), radius=4.5, dxfattribs={"layer": "VISIBLE"})

    rod_pts = [
        (px - 11, py - 4), (px - 14, py + 8),
        (cx - 6, pin_wrist_y - 12), (cx - 10, pin_wrist_y), (cx - 10, pin_wrist_y + 8),
        (cx + 10, pin_wrist_y + 8), (cx + 10, pin_wrist_y), (cx + 6, pin_wrist_y - 12),
        (px + 14, py + 8), (px + 11, py - 4)
    ]
    msp.add_lwpolyline(rod_pts, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((px - 14, py), radius=2.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((px + 14, py), radius=2.0, dxfattribs={"layer": "VISIBLE"})

    pw = 41.8
    piston_pts = [
        (cx - pw, 88), (cx - pw, 118),
        (cx - 25, 118), (cx - 15, 116), (cx + 15, 116), (cx + 25, 118),
        (cx + pw, 118), (cx + pw, 88),
        (cx + pw - 6, 88), (cx + pw - 6, 96), (cx - pw + 6, 96), (cx - pw + 6, 88)
    ]
    msp.add_lwpolyline(piston_pts, close=True, dxfattribs={"layer": "VISIBLE"})
    for ry in (114, 111, 108):
        msp.add_line((cx - pw, ry), (cx - pw + 3.5, ry), dxfattribs={"layer": "VISIBLE"})
        msp.add_line((cx + pw - 3.5, ry), (cx + pw, ry), dxfattribs={"layer": "VISIBLE"})

    bw = 42.0
    msp.add_line((cx - bw, 60), (cx - bw, 130), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx + bw, 60), (cx + bw, 130), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx - bw - 9, 60), (cx - bw - 9, 130), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx + bw + 9, 60), (cx + bw + 9, 130), dxfattribs={"layer": "VISIBLE"})

    for side in (-1, 1):
        wj = [(cx + side * (bw + 12), 75), (cx + side * (bw + 22), 75),
              (cx + side * (bw + 22), 125), (cx + side * (bw + 12), 125)]
        msp.add_lwpolyline(wj, close=True, dxfattribs={"layer": "HIDDEN"})

    msp.add_line((cx - 75, 130), (cx + 75, 130), dxfattribs={"layer": "VISIBLE"})

    msp.add_line((cx - bw, 130), (cx, 139), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx, 139), (cx + bw, 130), dxfattribs={"layer": "VISIBLE"})

    msp.add_line((cx - 4, 139), (cx - 4, 178), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx + 4, 139), (cx + 4, 178), dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, 182), radius=3.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx - 1.5, 136), (cx + 1.5, 136), dxfattribs={"layer": "VISIBLE"})

    for side, ang_deg, v_diam in [(-1, 15.0, 31.0), (1, -15.0, 27.0)]:
        rad = math.radians(ang_deg)
        vx = cx + side * 22.0
        vy = 135.0
        msp.add_line((vx - side * (v_diam/2), vy - 1), (vx + side * (v_diam/2), vy + 2), dxfattribs={"layer": "VISIBLE"})
        sx0, sy0 = vx, vy
        sx1, sy1 = vx - 36.0 * math.sin(rad), vy + 36.0 * math.cos(rad)
        msp.add_line((sx0 - 2, sy0), (sx1 - 2, sy1), dxfattribs={"layer": "VISIBLE"})
        msp.add_line((sx0 + 2, sy0), (sx1 + 2, sy1), dxfattribs={"layer": "VISIBLE"})
        for sp_y in np.linspace(vy + 10, sy1 - 5, 6):
            msp.add_line((sx0 - 7, sp_y), (sx0 + 7, sp_y + 2), dxfattribs={"layer": "VISIBLE"})

    intake_cam_x, intake_cam_y = cx - 32.0, 174.0
    exhaust_cam_x, exhaust_cam_y = cx + 32.0, 174.0
    msp.add_circle((intake_cam_x, intake_cam_y), radius=9.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((intake_cam_x, intake_cam_y), radius=4.5, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((exhaust_cam_x, exhaust_cam_y), radius=9.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((exhaust_cam_x, exhaust_cam_y), radius=4.5, dxfattribs={"layer": "VISIBLE"})

    msp.add_lwpolyline([(intake_cam_x - 9, intake_cam_y), (intake_cam_x - 4, intake_cam_y - 14),
                        (intake_cam_x + 4, intake_cam_y - 14), (intake_cam_x + 9, intake_cam_y)], close=False, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline([(exhaust_cam_x - 9, exhaust_cam_y), (exhaust_cam_x - 4, exhaust_cam_y - 14),
                        (exhaust_cam_x + 4, exhaust_cam_y - 14), (exhaust_cam_x + 9, exhaust_cam_y)], close=False, dxfattribs={"layer": "VISIBLE"})

    head_pts = [
        (cx - 65, 130), (cx - 68, 165), (cx - 48, 188), (cx - 15, 188),
        (cx + 15, 188), (cx + 48, 188), (cx + 68, 165), (cx + 65, 130)
    ]
    msp.add_lwpolyline(head_pts, close=False, dxfattribs={"layer": "VISIBLE"})

    msp.add_line((cx - bw, 122), (cx + bw, 122), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx - bw, 120), (cx - bw, 124), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx + bw, 120), (cx + bw, 124), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("BORE: 84.00 mm", height=2.2, dxfattribs={"layer": "DIMENSIONS"}).set_placement((cx - 18, 123.5))

    msp.add_line((45, crank_y - pin_r), (45, crank_y + pin_r), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((42, crank_y - pin_r), (48, crank_y - pin_r), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((42, crank_y + pin_r), (48, crank_y + pin_r), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("STROKE: 88.0 mm (44mm CRANK RADIUS)", height=2.0, dxfattribs={"layer": "DIMENSIONS"}).set_placement((15, 40))

    return doc

# =============================================================================
# 5. F-35 JET (Lockheed Martin F-35A Lightning II Stealth Fighter)
# =============================================================================
def build_f35_jet():
    doc, msp = setup_dxf("LOCKHEED MARTIN F-35A LIGHTNING II", "SF-F35-2026-10", "1:100")
    cx = 135.0

    msp.add_line((cx, 18), (cx, 188), dxfattribs={"layer": "CENTER"})

    nose_y = 184.0

    fuse_l = [
        (cx, nose_y), (cx - 5, 172), (cx - 10, 156), (cx - 16, 142),
        (cx - 24, 130),
        (cx - 22, 122), (cx - 20, 114)
    ]
    fuse_r = [(2*cx - p[0], p[1]) for p in fuse_l]
    msp.add_lwpolyline(fuse_l, close=False, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(fuse_r, close=False, dxfattribs={"layer": "VISIBLE"})

    canopy_pts = [
        (cx, 172), (cx - 5.5, 164), (cx - 6.8, 150), (cx - 5.2, 138),
        (cx, 136),
        (cx + 5.2, 138), (cx + 6.8, 150), (cx + 5.5, 164)
    ]
    msp.add_lwpolyline(canopy_pts, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx, 168), (cx, 142), dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, 148), radius=2.5, dxfattribs={"layer": "VISIBLE"})

    msp.add_lwpolyline([(cx - 24, 130), (cx - 15, 130), (cx - 15, 120), (cx - 22, 122)], close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline([(cx + 24, 130), (cx + 15, 130), (cx + 15, 120), (cx + 22, 122)], close=True, dxfattribs={"layer": "VISIBLE"})

    w_span_half = 53.5
    wing_l = [
        (cx - 20, 114),
        (cx - w_span_half, 82),
        (cx - w_span_half, 72),
        (cx - 24, 58),
        (cx - 18, 52)
    ]
    wing_r = [(2*cx - p[0], p[1]) for p in wing_l]
    msp.add_lwpolyline(wing_l, close=False, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(wing_r, close=False, dxfattribs={"layer": "VISIBLE"})

    msp.add_line((cx - 50, 78), (cx - 28, 64), dxfattribs={"layer": "VISIBLE"})
    msp.add_line((cx + 50, 78), (cx + 28, 64), dxfattribs={"layer": "VISIBLE"})

    vfin_l = [
        (cx - 16, 74), (cx - 28, 38), (cx - 26, 32), (cx - 14, 44)
    ]
    vfin_r = [(2*cx - p[0], p[1]) for p in vfin_l]
    msp.add_lwpolyline(vfin_l, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(vfin_r, close=True, dxfattribs={"layer": "VISIBLE"})

    htail_l = [
        (cx - 16, 50), (cx - 38, 26), (cx - 36, 18), (cx - 14, 30)
    ]
    htail_r = [(2*cx - p[0], p[1]) for p in htail_l]
    msp.add_lwpolyline(htail_l, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_lwpolyline(htail_r, close=True, dxfattribs={"layer": "VISIBLE"})

    nozzle_y = 30.0
    msp.add_circle((cx, nozzle_y), radius=8.0, dxfattribs={"layer": "VISIBLE"})
    msp.add_circle((cx, nozzle_y), radius=6.0, dxfattribs={"layer": "VISIBLE"})
    for i in range(12):
        pa = math.radians(i * 30.0)
        px0 = cx + 8.0 * math.cos(pa)
        py0 = nozzle_y + 8.0 * math.sin(pa)
        px1 = cx + 9.5 * math.cos(pa + math.radians(15.0))
        py1 = nozzle_y + 9.5 * math.sin(pa + math.radians(15.0))
        msp.add_line((px0, py0), (px1, py1), dxfattribs={"layer": "VISIBLE"})

    msp.add_lwpolyline([(cx - 15, 108), (cx - 7, 108), (cx - 7, 72), (cx - 15, 72)], close=True, dxfattribs={"layer": "HIDDEN"})
    msp.add_lwpolyline([(cx + 15, 108), (cx + 7, 108), (cx + 7, 72), (cx + 15, 72)], close=True, dxfattribs={"layer": "HIDDEN"})

    prof_y = 22.0
    msp.add_line((18, prof_y), (88, prof_y), dxfattribs={"layer": "CENTER"})
    side_pts = [
        (20, prof_y + 2), (28, prof_y + 4), (40, prof_y + 11), (50, prof_y + 8),
        (65, prof_y + 6), (78, prof_y + 14), (84, prof_y + 3), (85, prof_y),
        (75, prof_y - 2), (55, prof_y - 3), (35, prof_y - 1), (20, prof_y + 2)
    ]
    msp.add_lwpolyline(side_pts, close=True, dxfattribs={"layer": "VISIBLE"})
    msp.add_text("SIDE ELEVATION", height=1.8, dxfattribs={"layer": "TEXT"}).set_placement((35, prof_y - 7))

    msp.add_line((cx - w_span_half, 102), (cx + w_span_half, 102), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx - w_span_half, 98), (cx - w_span_half, 106), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx + w_span_half, 98), (cx + w_span_half, 106), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("WINGSPAN: 10.70 m [107.0 mm]", height=2.3, dxfattribs={"layer": "DIMENSIONS"}).set_placement((cx - 26, 104))

    msp.add_line((cx + 62, 30), (cx + 62, 184), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx + 58, 30), (cx + 66, 30), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((cx + 58, 184), (cx + 66, 184), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("LENGTH: 15.67 m [154.0 mm]", height=2.3, dxfattribs={"layer": "DIMENSIONS"}).set_placement((cx + 65, 110))

    return doc

def main():
    builders = [
        ("06_wheel_technical_drawing", build_wheel),
        ("07_butterfly_sketch", build_butterfly),
        ("08_cat_silhouette_sketch", build_cat),
        ("09_engine_technical_drawing", build_engine),
        ("10_f35_jet_technical_drawing", build_f35_jet),
    ]

    for filename_stem, builder in builders:
        print(f"\n--- Generating {filename_stem} ---")
        doc = builder()

        dxf_name = f"{filename_stem}.dxf"
        png_name = f"{filename_stem}.png"

        repo_dxf = os.path.join(TEST_DXF_DIR, dxf_name)
        repo_png = os.path.join(TEST_DXF_DIR, png_name)
        doc.saveas(repo_dxf)
        print(f"Saved DXF: {repo_dxf}")
        render_dxf_to_png(doc, repo_png)

        desktop_dxf = os.path.join(DESKTOP_DIR, dxf_name)
        desktop_png = os.path.join(DESKTOP_DIR, png_name)
        shutil.copyfile(repo_dxf, desktop_dxf)
        shutil.copyfile(repo_png, desktop_png)
        print(f"Copied to Desktop: {desktop_dxf} and {desktop_png}")

    print("\n[SUCCESS] All 5 requested DXF CAD models & technical drawing pictures generated!")

if __name__ == "__main__":
    main()
