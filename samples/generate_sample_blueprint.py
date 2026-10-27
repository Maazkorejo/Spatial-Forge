import cv2
import numpy as np

def create_synthetic_blueprint(output_path: str = "sample_bracket_blueprint.png"):
    # Create white canvas (mimicking clean drafting paper or scan)
    h, w = 800, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Color palette
    line_color = (30, 30, 30)      # Charcoal ink
    dim_color = (40, 40, 180)      # Drafting red/blue ink
    text_color = (20, 20, 20)

    # Base rectangle dimensions in pixels (representing 120mm x 84mm)
    bx, by, bw, bh = 180, 240, 440, 320
    radius = 35

    # Draw rounded rectangle (outer hull)
    # Top-left, top-right, bottom-right, bottom-left
    cv2.rectangle(img, (bx + radius, by), (bx + bw - radius, by + bh), line_color, 3)
    cv2.rectangle(img, (bx, by + radius), (bx + bw, by + bh - radius), line_color, 3)
    cv2.circle(img, (bx + radius, by + radius), radius, line_color, 3)
    cv2.circle(img, (bx + bw - radius, by + radius), radius, line_color, 3)
    cv2.circle(img, (bx + radius, by + bh - radius), radius, line_color, 3)
    cv2.circle(img, (bx + bw - radius, by + bh - radius), radius, line_color, 3)

    # Fill internal body with white to clear seam lines
    cv2.rectangle(img, (bx + radius - 2, by + 2), (bx + bw - radius + 2, by + bh - 2), (255, 255, 255), -1)
    cv2.rectangle(img, (bx + 2, by + radius - 2), (bx + bw - 2, by + bh - radius + 2), (255, 255, 255), -1)

    # Redraw clean outer border
    cv2.rectangle(img, (bx + radius, by), (bx + bw - radius, by + bh), line_color, 3)
    cv2.rectangle(img, (bx, by + radius), (bx + bw, by + bh - radius), line_color, 3)
    cv2.circle(img, (bx + radius, by + radius), radius, line_color, 3)
    cv2.circle(img, (bx + bw - radius, by + radius), radius, line_color, 3)
    cv2.circle(img, (bx + radius, by + bh - radius), radius, line_color, 3)
    cv2.circle(img, (bx + bw - radius, by + bh - radius), radius, line_color, 3)

    # Center Bore (Ø44.0mm)
    center_x, center_y = bx + bw // 2, by + bh // 2
    bore_r = 75
    cv2.circle(img, (center_x, center_y), bore_r, line_color, 3)
    # Center crosshair
    cv2.line(img, (center_x - 90, center_y), (center_x + 90, center_y), (180, 180, 180), 1)
    cv2.line(img, (center_x, center_y - 90), (center_x, center_y + 90), (180, 180, 180), 1)

    # 4 Corner Holes (M6)
    hole_offset = 60
    hole_r = 22
    corners = [
        (bx + hole_offset, by + hole_offset),
        (bx + bw - hole_offset, by + hole_offset),
        (bx + hole_offset, by + bh - hole_offset),
        (bx + bw - hole_offset, by + bh - hole_offset)
    ]
    for cx, cy in corners:
        cv2.circle(img, (cx, cy), hole_r, line_color, 2)

    # Dimension Callout Lines & Text
    # 1. Top dimension (120.0 mm)
    cv2.line(img, (bx, by - 40), (bx + bw, by - 40), dim_color, 2)
    cv2.line(img, (bx, by - 55), (bx, by - 25), dim_color, 2)
    cv2.line(img, (bx + bw, by - 55), (bx + bw, by - 25), dim_color, 2)
    cv2.putText(img, "120.0 mm", (bx + bw // 2 - 60, by - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

    # 2. Right dimension (84.0 mm)
    cv2.line(img, (bx + bw + 40, by), (bx + bw + 40, by + bh), dim_color, 2)
    cv2.line(img, (bx + bw + 25, by), (bx + bw + 55, by), dim_color, 2)
    cv2.line(img, (bx + bw + 25, by + bh), (bx + bw + 55, by + bh), dim_color, 2)
    cv2.putText(img, "84.0 mm", (bx + bw + 50, by + bh // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

    # 3. Center bore callout (Ø44.0)
    cv2.line(img, (center_x + 50, center_y - 50), (center_x + 130, center_y - 120), dim_color, 2)
    cv2.line(img, (center_x + 130, center_y - 120), (center_x + 220, center_y - 120), dim_color, 2)
    cv2.putText(img, "DIA 44.0", (center_x + 135, center_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # 4. Hole specification callout (4x M6)
    cv2.line(img, (bx + hole_offset + 15, by + hole_offset - 15), (bx + hole_offset + 70, by - 10), dim_color, 2)
    cv2.line(img, (bx + hole_offset + 70, by - 10), (bx + hole_offset + 160, by - 10), dim_color, 2)
    cv2.putText(img, "4x M6", (bx + hole_offset + 75, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

    cv2.imwrite(output_path, img)
    print(f"Sample blueprint saved to: {output_path}")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_bracket_blueprint.png"
    create_synthetic_blueprint(out)
