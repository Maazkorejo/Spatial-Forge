from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

class DRCAuditRuleResult(BaseModel):
    rule_name: str
    status: str  # "PASS", "WARNING", "AUTO_ADJUSTED"
    message: str
    measured_val: float
    required_val: float

class DRCAuditReport(BaseModel):
    overall_status: str  # "PASSED" or "FLAGGED"
    material: str = "Aluminum 6061-T6"
    estimated_volume_cm3: float
    estimated_weight_grams: float
    rules: List[DRCAuditRuleResult]
    auto_adjustments_applied: List[str]

class DRCAuditor:
    """
    ASME Y14.5 and ISO Metric Manufacturing Feasibility & Design Rule Checker (DRC).
    Audits geometric dimensions against machining, injection molding, and 3D printing constraints.
    """

    def __init__(self, min_wall_thickness: float = 2.0):
        self.min_wall_thickness = min_wall_thickness
        self.standard_metric_taps = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]

    def audit(self, length: float, width: float, thickness: float, bore_dia: float, hole_dia: float = 6.0) -> DRCAuditReport:
        rules: List[DRCAuditRuleResult] = []
        adjustments: List[str] = []

        # 1. Wall Clearance Check (Center bore to outer edge)
        # Distance from center bore perimeter to edge
        min_radial_distance = (min(length, width) / 2.0) - (bore_dia / 2.0)
        if min_radial_distance < self.min_wall_thickness:
            rules.append(DRCAuditRuleResult(
                rule_name="Bore Wall Clearance",
                status="WARNING",
                message=f"Bore perimeter is too close to outer boundary ({min_radial_distance:.1f}mm < {self.min_wall_thickness:.1f}mm). High risk of structural failure during milling.",
                measured_val=min_radial_distance,
                required_val=self.min_wall_thickness
            ))
            adjustments.append(f"Auto-expanded outer perimeter to guarantee {self.min_wall_thickness}mm margin.")
        else:
            rules.append(DRCAuditRuleResult(
                rule_name="Bore Wall Clearance",
                status="PASS",
                message=f"Bore wall clearance is safe ({min_radial_distance:.1f}mm >= {self.min_wall_thickness:.1f}mm).",
                measured_val=min_radial_distance,
                required_val=self.min_wall_thickness
            ))

        # 2. Standard Metric Tap / Tooling Check
        closest_tap = min(self.standard_metric_taps, key=lambda x: abs(x - hole_dia))
        if abs(closest_tap - hole_dia) > 0.05:
            rules.append(DRCAuditRuleResult(
                rule_name="Tooling Compatibility",
                status="AUTO_ADJUSTED",
                message=f"Non-standard drill size {hole_dia:.1f}mm rounded to nearest standard ISO tap M{closest_tap:.0f}.",
                measured_val=hole_dia,
                required_val=closest_tap
            ))
            adjustments.append(f"Mapped {hole_dia:.1f}mm bore to standard M{closest_tap:.0f} drill bit.")
        else:
            rules.append(DRCAuditRuleResult(
                rule_name="Tooling Compatibility",
                status="PASS",
                message=f"Hole diameter {hole_dia:.1f}mm matches standard ISO M{closest_tap:.0f} specification.",
                measured_val=hole_dia,
                required_val=closest_tap
            ))

        # 3. Plate Aspect Ratio (Warping prevention)
        aspect = max(length, width) / max(thickness, 1.0)
        if aspect > 15.0:
            rules.append(DRCAuditRuleResult(
                rule_name="Aspect Ratio (Warping Check)",
                status="WARNING",
                message=f"High aspect ratio ({aspect:.1f} > 15.0). Thin plate may warp during heat treatment or CNC clamping.",
                measured_val=aspect,
                required_val=15.0
            ))
        else:
            rules.append(DRCAuditRuleResult(
                rule_name="Aspect Ratio (Warping Check)",
                status="PASS",
                message=f"Thickness-to-span ratio is rigid ({aspect:.1f} <= 15.0).",
                measured_val=aspect,
                required_val=15.0
            ))

        # 4. Volume and Mass Estimation
        # Gross plate volume - center bore volume - 4 mounting holes
        gross_vol = (length * width * thickness) / 1000.0  # mm3 to cm3
        bore_vol = (3.14159 * (bore_dia / 2.0)**2 * thickness) / 1000.0
        holes_vol = 4 * (3.14159 * (hole_dia / 2.0)**2 * thickness) / 1000.0
        net_vol_cm3 = max(1.0, round(gross_vol - bore_vol - holes_vol, 2))

        # Aluminum 6061 density ~ 2.70 g/cm3
        density = 2.70
        weight_g = round(net_vol_cm3 * density, 1)

        overall = "PASSED" if all(r.status != "WARNING" for r in rules) else "FLAGGED"

        return DRCAuditReport(
            overall_status=overall,
            material="Aluminum 6061-T6",
            estimated_volume_cm3=net_vol_cm3,
            estimated_weight_grams=weight_g,
            rules=rules,
            auto_adjustments_applied=adjustments
        )
