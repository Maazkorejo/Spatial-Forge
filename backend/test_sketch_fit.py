import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spatialforge_v2", "spatialforge_v2")))

import fit as F
from examples import sewing_machine
from solver import resolve

img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "sewing_machine_sketch.jpg"))
img_bgr = cv2.imread(img_path)

print(f"Loaded image: {img_path}, shape: {img_bgr.shape}")

# Extract mask
from pipeline import extract_mask
mask = extract_mask(img_bgr)
cv2.imwrite("test_extracted_mask.png", mask)
print(f"Extracted mask saved to test_extracted_mask.png, nonzero pixels: {cv2.countNonZero(mask)}")

target = F.Target(mask, S=160)
asm = sewing_machine()

print("Searching camera...")
cam = F.search_camera(asm, target)
print(f"Initial Camera: az={cam.az:.1f}, el={cam.el:.1f}")

print("Fitting continuous parameters against sketch silhouette...")
fr = F.fit(asm, target, cam=cam)
print(f"Fit complete! IoU {fr.init_iou:.3f} -> {fr.iou:.3f}")
print(f"Camera refined: az={fr.camera.az:.1f}, el={fr.camera.el:.1f}")

fitted_asm = F.apply_fit(asm, fr)
overlay = F.make_overlay(fr.rendered, target)
with open("test_fitted_overlay.png", "wb") as f:
    f.write(overlay)

print("Saved test_fitted_overlay.png!")
for p in fitted_asm.parts:
    for f_name, d in p.dims().items():
        print(f"  {p.id}.{f_name}: {d.value:.2f} ({d.source})")
