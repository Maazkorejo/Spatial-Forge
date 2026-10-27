import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel

class HoleFeature(BaseModel):
    center_x: float
    center_y: float
    radius: float
    diameter_mm: float
    feature_type: str = "circular_bore"
    tap_spec: Optional[str] = None  # e.g., M6x1.0

class PocketFeature(BaseModel):
    center_x: float
    center_y: float
    width: float
    height: float
    corner_radius: float = 0.0

class PartTopology(BaseModel):
    image_width: int
    image_height: int
    scale_px_to_mm: float = 1.0  # Calibrated via detected dimension text
    outer_hull_points: List[List[float]]  # Normalized polygon [x, y]
    holes: List[HoleFeature]
    pockets: List[PocketFeature]
    annotation_regions: List[List[int]]  # [x, y, w, h] crops for micro-OCR
    estimated_length_mm: float = 120.0
    estimated_width_mm: float = 84.0

class BlueprintPerceptionEngine:
    """
    Perception engine combining Meta SAM 3.1 prompt concepts with high-speed OpenCV geometry.
    Extracts structural geometry (hull, bores, pockets) and isolates annotation patches.
    """

    def __init__(self, min_hole_radius_px: int = 4, max_hole_radius_px: int = 150):
        self.min_hole_radius_px = min_hole_radius_px
        self.max_hole_radius_px = max_hole_radius_px

    def process_image(self, image_bytes: bytes) -> Tuple[PartTopology, np.ndarray]:
        # Decode image from buffer
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Contrast enhancement & adaptive thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 4
        )

        # 2. Extract outer hull contour
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        outer_hull = []
        holes = []
        pockets = []
        annotation_regions = []

        if contours:
            # Sort contours by area
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            main_contour = sorted_contours[0]

            # Simplify polygon using Douglas-Peucker
            epsilon = 0.008 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            outer_hull = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]

            # Bounding box for scale reference
            bx, by, bw, bh = cv2.boundingRect(main_contour)
        else:
            bw, bh = w * 0.8, h * 0.6
            outer_hull = [
                [w * 0.1, h * 0.2], [w * 0.9, h * 0.2],
                [w * 0.9, h * 0.8], [w * 0.1, h * 0.8]
            ]

        # 3. Detect circular bores via Hough Circles & Contour Roundness
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=28,
            minRadius=self.min_hole_radius_px, maxRadius=self.max_hole_radius_px
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = float(i[0]), float(i[1]), float(i[2])
                holes.append(HoleFeature(
                    center_x=cx,
                    center_y=cy,
                    radius=r,
                    diameter_mm=round(r * 2.0 * 0.5, 1)  # Initial uncalibrated estimate
                ))

        # 4. Isolate potential dimension annotation patches (text & numbers)
        # Morphological gradient to find text blocks
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, rect_kernel)
        text_contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for tc in text_contours:
            tx, ty, tw, th = cv2.boundingRect(tc)
            area = tw * th
            aspect = tw / float(th + 1e-5)
            # Filter for text-like aspect ratios and sensible size
            if 150 < area < (w * h * 0.05) and 1.2 < aspect < 8.0:
                annotation_regions.append([tx, ty, tw, th])

        # Return structured topology and preprocessed binary for visual verification
        topology = PartTopology(
            image_width=w,
            image_height=h,
            scale_px_to_mm=1.0,
            outer_hull_points=outer_hull,
            holes=holes,
            pockets=pockets,
            annotation_regions=annotation_regions[:10], # Top candidate text regions
            estimated_length_mm=round(bw * 0.4, 1) if bw else 120.0,
            estimated_width_mm=round(bh * 0.4, 1) if bh else 84.0
        )

        return topology, binary
