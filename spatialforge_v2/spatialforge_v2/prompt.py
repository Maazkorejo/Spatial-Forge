"""
Constrained-output prompts  (v2)

Use together with schema-constrained decoding where available:
  * Ollama:  pass `format=schema.json_schema()`
  * OpenAI-compatible (Groq etc.): `response_format={"type":"json_schema", ...}` if the model
    supports it, else `json_object` mode. Either way, validate with schema.parse_assembly()
    and feed errors back (see pipeline.propose).
"""
from __future__ import annotations

import json
from typing import Optional

from examples import sewing_machine
from schema import Assembly, json_schema

SYSTEM_PROMPT = """You are a mechanical-design assistant. You look at a sketch or drawing of a physical object and \
decompose it into a small assembly of 3D primitives, written in a strict JSON DSL.

YOUR JOB IS STRUCTURE, NOT PRECISE MEASUREMENT.
A separate optimizer will fit the numbers to the image afterwards. You decide WHAT the parts are and HOW they \
connect. You give dimensions only as rough proportions with honest uncertainty ranges.

RULES
1. Output ONE JSON object that matches the schema exactly. No prose, no markdown fences, no extra keys.
2. Fill "observations" first (<=8 short factual notes: what the object is, the viewing angle, the main components). \
Then build the parts from those notes.
3. Kinds: box, rounded_box (needs fillet > 0), cylinder (needs axis), sphere, extrusion. Use at most 12 parts.
   Use "extrusion" for a body whose SIDE outline is not a rectangle (sloped nose, notch, C or L shape): give "profile"
   as a simple polygon in the X-Z plane, counter-clockwise, 3-24 points [x, z] in 0..1 of the part's width_x/height_z
   box; it is extruded along y by depth_y. Prefer rounded_box over box for visibly rounded bodies.
   Do NOT output buttons, dials, screws, text, slots or other small details: a separate detector places them.
   Leave "features" out entirely.
4. Ignore everything that is not the object: construction/guide lines, vanishing lines, hatching, shadows, \
paper edges, annotation text (except real dimension labels, see rule 8).
5. Never use absolute coordinates. The root part (placement = null) sits at the origin. Every other part is attached \
to a face of its parent ("top","bottom","left","right","front","back") with optional alignment/offset and "embed". \
Axes: x = right, y = back (away from a viewer facing the front), z = up.
6. Units are arbitrary "u". Scale the object so its longest overall extent is about 100u.
7. Every dimension is {value, min, max, source}. Make ranges generous (about +-30%) unless the dimension is printed \
in the image (then +-5%).
8. source: "measured" ONLY if you read the number from a printed dimension in the image (then convert so it is \
consistent with the 100u scale, and also set "anchor" with the real mm value). "inferred" = a proportion you can \
see. "assumed" = hidden or guessed. When unsure, choose the lower-confidence label.
9. Cavities, holes, slots: model as op="difference" with "cuts": [ids of parts they remove material from]. \
A cut must actually overlap its target (use "embed" ~0.5-0.9).
10. If you cannot tell whether a part exists, OMIT it and lower "confidence". Never invent parts to look complete.
11. Orientation sanity: heavy bases sit at the bottom, supported parts sit on top of their supports. Prefer \
symmetric, axis-aligned arrangements unless the image clearly shows otherwise.
12. "view_hint": give your best estimate of projection and azimuth/elevation (azimuth 0 = looking at the front, \
positive = camera moved toward +x). Use null if unsure.

SCHEMA (JSON Schema):
@@SCHEMA@@

EXAMPLE (a different drawing of a similar kind of object; do not copy it, adapt to the image):
@@EXAMPLE@@
"""


def build_system_prompt() -> str:
    ex = sewing_machine().model_dump_json(exclude_none=True, exclude={"features"})
    return (SYSTEM_PROMPT.replace("@@SCHEMA@@", json.dumps(json_schema(), separators=(",", ":")))
            .replace("@@EXAMPLE@@", ex))


def build_user_prompt(anchor_hint: Optional[str] = None) -> str:
    s = "Decompose the object in the image into the JSON DSL. Return JSON only."
    if anchor_hint:
        s += f" The user states: {anchor_hint}. Use it to set 'anchor' (source='measured')."
    return s


def build_error_feedback(errors: list[str]) -> str:
    """Appended to the next attempt when the previous output failed schema/geometry validation."""
    lines = "\n".join(f"- {e}" for e in errors[:10])
    return ("Your previous output was rejected by the validator:\n" + lines +
            "\nReturn a corrected, complete JSON object. JSON only.")


def build_repair_prompt(asm: Assembly, iou: float, diag: dict, weak_parts: list[str],
                        view_issues: Optional[list] = None) -> str:
    """Structural repair after the render-compare loop. Images: [original, overlay].
    Overlay colours: green = sketch silhouette not covered by your model, red = your model
    sticking out of the sketch, yellow = agreement."""
    cur = asm.model_dump_json(exclude_none=True, exclude={"features"})
    msg = [
        f"Your assembly was rendered from the estimated camera and compared with the sketch silhouette "
        f"(IoU = {iou:.2f}, target >= 0.85).",
        "Image 1 is the original sketch. Image 2 is the overlay: GREEN = silhouette you did not explain, "
        "RED = your model sticks out where the sketch has nothing, YELLOW = agreement.",
        f"Uncovered silhouette: {diag['uncovered_frac']:.0%} of the sketch, centred near {diag['uncovered_center']} "
        f"(x,y in 0..1, origin top-left).",
        f"Excess model: {diag['excess_frac']:.0%}, centred near {diag['excess_center']}.",
    ]
    if weak_parts:
        msg.append("Parts the silhouette cannot constrain (check they are real): " + ", ".join(weak_parts) + ".")
    if view_issues:
        msg.append("A reviewer looking at renders of your model from other angles reported: " + "; ".join(view_issues) + ".")
    msg += [
        "Fix STRUCTURE only: add a missing part, remove a part that does not exist, change a primitive kind, "
        "an attachment face, an alignment, or an axis. Do NOT tune numbers to chase the overlay; the optimizer "
        "handles dimensions. Keep everything that already agrees (yellow).",
        "Current assembly:", cur,
        "Return the complete corrected JSON object. JSON only.",
    ]
    return "\n".join(msg)
