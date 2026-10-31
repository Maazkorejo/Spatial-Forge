"""
SpatialForge - 50 CAD Test Dataset Generator
Generates 50 production-grade 2D CAD technical drawings (.dxf) and high-res blueprints (.png)
spanning Hand Tools, Mechanical Objects, and Complex/Difficult Engineering CADs.
"""
import os
import sys
import math
import numpy as np
import ezdxf
import cv2

OUTPUT_DIR = os.path.abspath("test_cad_models")
DESKTOP_DIR = os.path.expanduser(r"~\Desktop\Spatial Forge Test CADs")

os.makedirs(os.path.join(OUTPUT_DIR, "01_tools"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "02_objects"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "03_difficult_cads"), exist_ok=True)

os.makedirs(os.path.join(DESKTOP_DIR, "01_tools"), exist_ok=True)
os.makedirs(os.path.join(DESKTOP_DIR, "02_objects"), exist_ok=True)
os.makedirs(os.path.join(DESKTOP_DIR, "03_difficult_cads"), exist_ok=True)

LAYERS = {
    "VISIBLE": ("CONTINUOUS", 7),
    "HIDDEN": ("DASHED", 8),
    "CENTER": ("CENTER", 1),
    "DIM": ("CONTINUOUS", 3),
    "TITLE": ("CONTINUOUS", 5),
}

def create_cad_doc():
    doc = ezdxf.new("R2010", setup=True)
    doc.units = 4  # Millimeters
    for name, (lt, col) in LAYERS.items():
        doc.layers.add(name, color=col, linetype=lt)
    return doc

def add_lines(msp, pts, closed=True, layer="VISIBLE", ltype="CONTINUOUS"):
    n = len(pts)
    count = n if closed else n - 1
    for i in range(count):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        msp.add_line(p1, p2, dxfattribs={"layer": layer, "linetype": ltype})

def add_rect(msp, x, y, w, h, layer="VISIBLE", ltype="CONTINUOUS"):
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    add_lines(msp, pts, closed=True, layer=layer, ltype=ltype)

def add_circle(msp, cx, cy, r, layer="VISIBLE", centerlines=True):
    msp.add_circle((cx, cy), r, dxfattribs={"layer": layer})
    if centerlines:
        ext = r + 4.0
        msp.add_line((cx - ext, cy), (cx + ext, cy), dxfattribs={"layer": "CENTER", "linetype": "CENTER"})
        msp.add_line((cx, cy - ext), (cx, cy + ext), dxfattribs={"layer": "CENTER", "linetype": "CENTER"})

def add_arc(msp, cx, cy, r, a0, a1, layer="VISIBLE"):
    msp.add_arc((cx, cy), r, a0, a1, dxfattribs={"layer": layer})

def add_slot(msp, cx, cy, length, width, angle=0, layer="VISIBLE"):
    r = width / 2.0
    dx = (length - width) / 2.0
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    def rot(px, py):
        return (cx + px * c - py * s, cy + px * s + py * c)
    p1 = rot(-dx, r)
    p2 = rot(dx, r)
    p3 = rot(dx, -r)
    p4 = rot(-dx, -r)
    msp.add_line(p1, p2, dxfattribs={"layer": layer})
    msp.add_line(p3, p4, dxfattribs={"layer": layer})
    msp.add_arc(rot(dx, 0), r, angle - 90, angle + 90, dxfattribs={"layer": layer})
    msp.add_arc(rot(-dx, 0), r, angle + 90, angle + 270, dxfattribs={"layer": layer})

def add_dim(msp, p1, p2, offset, text, vertical=False):
    x1, y1 = p1
    x2, y2 = p2
    if vertical:
        x_dim = x1 + offset
        msp.add_line((x1, y1), (x_dim, y1), dxfattribs={"layer": "DIM"})
        msp.add_line((x2, y2), (x_dim, y2), dxfattribs={"layer": "DIM"})
        msp.add_line((x_dim, y1), (x_dim, y2), dxfattribs={"layer": "DIM"})
        msp.add_text(text, height=3.2, dxfattribs={"layer": "DIM", "insert": (x_dim + 1.5, (y1 + y2) / 2.0 - 1.5)})
    else:
        y_dim = y1 + offset
        msp.add_line((x1, y1), (x1, y_dim), dxfattribs={"layer": "DIM"})
        msp.add_line((x2, y2), (x2, y_dim), dxfattribs={"layer": "DIM"})
        msp.add_line((x1, y_dim), (x2, y_dim), dxfattribs={"layer": "DIM"})
        msp.add_text(text, height=3.2, dxfattribs={"layer": "DIM", "insert": ((x1 + x2) / 2.0 - len(text) * 1.0, y_dim + 1.5)})

def add_title_block(msp, title, part_no, bbox_dims, scale="1:1"):
    bx, by, bw, bh = bbox_dims
    tb_x = bx + bw - 110
    tb_y = by - 35
    add_rect(msp, tb_x, tb_y, 110, 28, layer="TITLE")
    msp.add_line((tb_x, tb_y + 14), (tb_x + 110, tb_y + 14), dxfattribs={"layer": "TITLE"})
    msp.add_line((tb_x + 65, tb_y), (tb_x + 65, tb_y + 14), dxfattribs={"layer": "TITLE"})
    msp.add_text(title[:26], height=3.8, dxfattribs={"layer": "TITLE", "insert": (tb_x + 4, tb_y + 18)})
    msp.add_text(f"PART: {part_no}", height=2.6, dxfattribs={"layer": "TITLE", "insert": (tb_x + 4, tb_y + 8)})
    msp.add_text("SPATIALFORGE 2D CAD", height=2.4, dxfattribs={"layer": "TITLE", "insert": (tb_x + 4, tb_y + 2)})
    msp.add_text(f"SCALE: {scale}", height=2.6, dxfattribs={"layer": "TITLE", "insert": (tb_x + 70, tb_y + 8)})
    msp.add_text("UNITS: MM", height=2.4, dxfattribs={"layer": "TITLE", "insert": (tb_x + 70, tb_y + 2)})

def render_blueprint(doc, out_png, margin=40, ppm=3.5):
    """Renders DXF document to clean, high-contrast engineering drawing PNG."""
    msp = doc.modelspace()
    lines = []
    circles = []
    arcs = []
    texts = []
    all_pts = []

    for e in msp:
        t = e.dxftype()
        lay = e.dxf.get("layer", "VISIBLE")
        lt = e.dxf.get("linetype", "CONTINUOUS")
        if t == "LINE":
            p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
            lines.append((p1, p2, lay, lt))
            all_pts.extend([p1, p2])
        elif t == "CIRCLE":
            c = (float(e.dxf.center.x), float(e.dxf.center.y))
            r = float(e.dxf.radius)
            circles.append((c, r, lay))
            all_pts.extend([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)])
        elif t == "ARC":
            c = (float(e.dxf.center.x), float(e.dxf.center.y))
            r = float(e.dxf.radius)
            arcs.append((c, r, float(e.dxf.start_angle), float(e.dxf.end_angle), lay))
            all_pts.extend([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)])
        elif t in ("TEXT", "MTEXT"):
            ins = (float(e.dxf.insert.x), float(e.dxf.insert.y))
            txt = e.dxf.text if t == "TEXT" else e.text
            h = float(e.dxf.height) if e.dxf.hasattr("height") else 3.0
            texts.append((ins, txt, h, lay))
            all_pts.append(ins)

    if not all_pts:
        return

    pts_arr = np.array(all_pts)
    min_x, min_y = pts_arr.min(axis=0)
    max_x, max_y = pts_arr.max(axis=0)

    w_mm = max(10, max_x - min_x)
    h_mm = max(10, max_y - min_y)
    
    img_w = int(w_mm * ppm) + 2 * margin
    img_h = int(h_mm * ppm) + 2 * margin

    # Blueprint theme: Slate charcoal background with crisp technical linework
    img = np.full((img_h, img_w, 3), (235, 238, 242), dtype=np.uint8)

    # Subtle grid
    grid_spacing = int(10.0 * ppm)
    if grid_spacing > 8:
        for gx in range(margin % grid_spacing, img_w, grid_spacing):
            cv2.line(img, (gx, 0), (gx, img_h), (220, 225, 232), 1)
        for gy in range(margin % grid_spacing, img_h, grid_spacing):
            cv2.line(img, (0, gy), (img_w, gy), (220, 225, 232), 1)

    def tx(x, y):
        px = int((x - min_x) * ppm + margin)
        py = int(img_h - ((y - min_y) * ppm + margin))
        return px, py

    COLOR_MAP = {
        "VISIBLE": (35, 38, 45),     # Deep charcoal visible edges
        "HIDDEN": (130, 90, 60),     # Dashed hidden lines
        "CENTER": (45, 75, 180),     # Reddish-blue centerlines
        "DIM": (40, 115, 60),        # Engineering green dimensions
        "TITLE": (60, 68, 80),       # Title block
    }

    # Draw centerlines first
    for p1, p2, lay, lt in lines:
        if lay == "CENTER":
            col = COLOR_MAP.get(lay, (45, 75, 180))
            a, b = tx(*p1), tx(*p2)
            cv2.line(img, a, b, col, 1, cv2.LINE_AA)

    # Draw hidden lines
    for p1, p2, lay, lt in lines:
        if lay == "HIDDEN":
            col = COLOR_MAP.get(lay, (130, 90, 60))
            a, b = tx(*p1), tx(*p2)
            # dashed line
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                steps = max(2, int(dist / 8))
                for s in range(steps):
                    if s % 2 == 0:
                        pt1 = (int(a[0] + dx * s / steps), int(a[1] + dy * s / steps))
                        pt2 = (int(a[0] + dx * min(s + 1, steps) / steps), int(a[1] + dy * min(s + 1, steps) / steps))
                        cv2.line(img, pt1, pt2, col, 1, cv2.LINE_AA)

    # Draw visible and title lines
    for p1, p2, lay, lt in lines:
        if lay not in ("CENTER", "HIDDEN"):
            col = COLOR_MAP.get(lay, (35, 38, 45))
            thick = 2 if lay == "VISIBLE" else 1
            a, b = tx(*p1), tx(*p2)
            cv2.line(img, a, b, col, thick, cv2.LINE_AA)

    # Draw circles
    for c, r, lay in circles:
        col = COLOR_MAP.get(lay, (35, 38, 45))
        thick = 2 if lay == "VISIBLE" else 1
        center_px = tx(*c)
        r_px = max(1, int(r * ppm))
        cv2.circle(img, center_px, r_px, col, thick, cv2.LINE_AA)

    # Draw arcs
    for c, r, a0, a1, lay in arcs:
        col = COLOR_MAP.get(lay, (35, 38, 45))
        thick = 2 if lay == "VISIBLE" else 1
        center_px = tx(*c)
        r_px = max(1, int(r * ppm))
        cv2.ellipse(img, center_px, (r_px, r_px), 0, -a1, -a0, col, thick, cv2.LINE_AA)

    # Draw texts
    for ins, txt, h, lay in texts:
        col = COLOR_MAP.get(lay, (35, 38, 45))
        pt = tx(*ins)
        font_scale = max(0.3, h * ppm / 30.0)
        clean_txt = txt.replace("Ø", "D").replace("±", "+-")
        cv2.putText(img, clean_txt, pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1, cv2.LINE_AA)

    cv2.imwrite(out_png, img)

print("Framework initialized successfully.")


def generate_all_50():
    from cad_definitions import ALL_BUILDERS
    import shutil

    catalog = []

    print(f"Generating {len(ALL_BUILDERS)} Production 2D CAD Drawings & DXF Models...")

    for idx, (category, file_stem, builder) in enumerate(ALL_BUILDERS, start=1):
        doc, title, part_no, bbox = builder()
        msp = doc.modelspace()
        add_title_block(msp, title, part_no, bbox)

        # Paths
        cat_dir = os.path.join(OUTPUT_DIR, category)
        dxf_path = os.path.join(cat_dir, f"{file_stem}.dxf")
        png_path = os.path.join(cat_dir, f"{file_stem}.png")

        # Save DXF
        doc.saveas(dxf_path)

        # Render Blueprint Image
        render_blueprint(doc, png_path)

        # Copy to Desktop Folder as well
        desk_cat_dir = os.path.join(DESKTOP_DIR, category)
        shutil.copy2(dxf_path, os.path.join(desk_cat_dir, f"{file_stem}.dxf"))
        shutil.copy2(png_path, os.path.join(desk_cat_dir, f"{file_stem}.png"))

        catalog.append({
            "index": idx,
            "category": category,
            "stem": file_stem,
            "title": title,
            "part_no": part_no,
            "dxf_path": dxf_path,
            "png_path": png_path,
            "bbox": bbox
        })
        print(f"[{idx:02d}/50] OK: {category}/{file_stem} -> {title}")

    # Write INDEX.md in test_cad_models
    index_md = os.path.join(OUTPUT_DIR, "INDEX.md")
    with open(index_md, "w", encoding="utf-8") as f:
        f.write("# SpatialForge - 50 2D CAD Test Dataset\n\n")
        f.write("A comprehensive test suite of 50 mechanical 2D CAD technical drawings (.dxf) and high-resolution blueprints (.png).\n\n")
        f.write("### Usage with SpatialForge\n")
        f.write("Drag and drop any `.dxf` or `.png` file directly onto the SpatialForge intake dropzone at `http://127.0.0.1:3000/` or `http://127.0.0.1:8085/` to reconstruct the 3D solid model.\n\n")
        f.write("| # | Category | File Name | Part Description | Part No | Width x Height (mm) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for item in catalog:
            bx, by, bw, bh = item["bbox"]
            f.write(f"| {item['index']:02d} | `{item['category']}` | `{item['stem']}.dxf` | {item['title']} | `{item['part_no']}` | {bw:.0f} × {bh:.0f} mm |\n")

    # Also write on Desktop
    shutil.copy2(index_md, os.path.join(DESKTOP_DIR, "INDEX.md"))
    print(f"\nSuccessfully generated 50 DXFs and 50 PNG technical drawings in:")
    print(f"  1. {OUTPUT_DIR}")
    print(f"  2. {DESKTOP_DIR}")

if __name__ == "__main__":
    generate_all_50()
