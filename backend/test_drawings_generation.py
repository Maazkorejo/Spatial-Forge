import os
import sys
import shutil

# Ensure backend and drawing_gen and spatialforge_v2 are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "spatialforge_v2", "spatialforge_v2"))
sys.path.insert(0, os.path.join(BASE_DIR, "drawing_gen", "drawing_gen"))

import cadquery as cq
from draw import build_drawing
from dxf_io import write_dxf
from preview import render_dxf
from dxf_pipeline import process_dxf_bytes
from reconstruction import (
    reconstruct_wheel,
    reconstruct_butterfly,
    reconstruct_cat,
    reconstruct_engine,
    reconstruct_f35_jet,
)

def generate_all_samples():
    output_dir = os.path.join(BASE_DIR, "test_dxf_drawings")
    desktop_dir = r"C:\Users\maazk\Desktop\DXF_Test_Samples"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(desktop_dir, exist_ok=True)

    items = [
        ("06_wheel_technical_drawing", reconstruct_wheel, "wheel", "FORGED ALLOY WHEEL"),
        ("07_butterfly_technical_drawing", reconstruct_butterfly, "butterfly", "MONARCH BUTTERFLY"),
        ("08_cat_technical_drawing", reconstruct_cat, "cat", "STYLIZED CAT FIGURINE"),
        ("09_engine_technical_drawing", reconstruct_engine, "engine", "INTERNAL COMBUSTION ENGINE"),
        ("10_f35_jet_technical_drawing", reconstruct_f35_jet, "f35_jet", "F-35A LIGHTNING II JET"),
    ]

    results = []

    for fname, reconstructor, family, title in items:
        print(f"\n==================================================")
        print(f"Generating: {title} ({fname})")
        print(f"==================================================")
        dummy_registered = {"dimensions": {}, "texts": [{"text": title}], "filename": fname}
        solid, info = reconstructor(dummy_registered)
        print(f"  Solid: Valid={solid.val().isValid()}, Volume={solid.val().Volume():.1f} mm3")

        # Generate ASME 3rd-angle orthographic multi-view drawing
        spec = {"family": family}
        dwg = build_drawing(solid, spec, standard="third", scale="1:1", name=title)

        dxf_path = os.path.join(output_dir, f"{fname}.dxf")
        png_path = os.path.join(output_dir, f"{fname}.png")

        write_dxf(dwg, dxf_path)
        render_dxf(dxf_path, png_path)

        # Copy to Desktop folder
        shutil.copy2(dxf_path, os.path.join(desktop_dir, f"{fname}.dxf"))
        shutil.copy2(png_path, os.path.join(desktop_dir, f"{fname}.png"))
        print(f"  Wrote DXF: {dxf_path}")
        print(f"  Wrote PNG: {png_path}")

        # Ingest and reconstruct through the full production pipeline
        with open(dxf_path, "rb") as f:
            dxf_bytes = f.read()
        res = process_dxf_bytes(dxf_bytes, filename=f"{fname}.dxf")
        status = res.get("status")
        rep = res.get("reprojection") or {}
        f1 = rep.get("mean_vis_f1", 0.0)
        obj_class = res.get("object_class")
        flagged = rep.get("flagged", True)

        print(f"  Result -> Status: {status}, Object: {obj_class}, Edge F1: {f1*100:.1f}%, Flagged: {flagged}")
        results.append({
            "name": title,
            "filename": fname,
            "status": status,
            "object_class": obj_class,
            "f1": f1,
            "flagged": flagged,
            "volume": solid.val().Volume()
        })

    summary_lines = [
        "================================================================================",
        "SPATIAL FORGE - 2D CAD TECHNICAL DRAWING & 3D B-REP GENERATION SUITE",
        "================================================================================",
        "",
        "The following 5 complex engineering drawings and authentic 3D CAD B-Rep solids",
        "have been generated in ASME 3rd-Angle Orthographic Multi-View projection and",
        "verified through OpenCascade HLR Ray-Cast Reprojection Verification:",
        "",
    ]
    for r in results:
        status_tag = "[PASS - OK]" if r["status"] == "ok" else "[FLAGGED]"
        summary_lines.append(f"  * {r['name']:30} {status_tag} | Edge F1: {r['f1']*100:.1f}% | 3D Vol: {r['volume']:10.1f} mm3")
        summary_lines.append(f"    - DXF Drawing: {r['filename']}.dxf")
        summary_lines.append(f"    - PNG Picture: {r['filename']}.png")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(os.path.join(output_dir, "SUMMARY_FIVE_MODELS.txt"), "w") as f:
        f.write(summary_text)
    with open(os.path.join(desktop_dir, "SUMMARY_FIVE_MODELS.txt"), "w") as f:
        f.write(summary_text)

if __name__ == "__main__":
    generate_all_samples()
