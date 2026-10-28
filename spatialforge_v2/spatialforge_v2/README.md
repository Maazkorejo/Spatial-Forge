# SpatialForge v2 — sketch -> layered, labelled, editable CAD

Layers: **body** (primitives, fitted) -> **features** (detected + back-projected) -> **cosmetic** (library parts).
Every dimension carries provenance: `measured | inferred | assumed`.

## Quick start
    pip install -r requirements.txt
    python test_all.py     # 25 checks, ~1 min, no GPU / VLM needed
    python demo.py out     # synthetic end-to-end run; writes STL, GLB, STEP, JSON, PNGs

## Modules
| file | role |
|---|---|
| schema.py | Pydantic DSL: box, rounded_box, cylinder, sphere, extrusion(profile); relational placement; features; ranges + provenance |
| solver.py | relations -> world-space solids; geometry validity checks |
| geometry.py | watertight meshes (convex hulls, ear-clipped extrusions), feature library |
| fit.py | render-compare fit (silhouette IoU + edge chamfer), camera search, sensitivity -> `assumed` labels |
| raster.py | numpy z-buffer: per-part visible masks, shaded views |
| features.py | detection -> ray cast -> (part, face, u, v) + foreshortening-corrected size; feature poses |
| multiview.py | renders other angles, asks the VLM whether the model still looks like the object |
| prompt.py / vlm.py | constrained-output prompts; Ollama + OpenAI-compatible (Groq) adapters; mock |
| pipeline.py | orchestration: N candidates -> validate -> fit -> repair -> features -> view check |
| export.py / cad_export.py | manifold3d STL + GLB; CadQuery STEP (true fillets / cylinders) |
| evaluate.py | part F1, silhouette IoU, dim error, validity, feature P/R |
| synth.py | synthetic sketches with known ground truth |

## Plugging in real models (not done in this package)
* `mask`: SAM / SAM 2 whole-object mask (extract_mask() is a crude fallback).
* `detector(image_bgr) -> list[features.Detection]`: wrap Florence-2 / GroundingDINO / OWL-ViT. Boxes in original pixels.
* `vlm`: `OllamaVLM` or `OpenAICompatVLM`; verify model ids and free-tier availability yourself.

## What is verified (test_all.py, synthetic data)
Schema rejects bad profiles/features; all meshes watertight; fit raises IoU 0.76 -> 0.97 and recovers the camera
within ~3 deg; 5/5 features land on the right part/face (position error <= 0.11 in face units, sizes within +-17%);
per-part IoU separates right from wrong structure; STL watertight; STEP valid, re-importable, volume within 0.2% of
the mesh; retry path survives invalid VLM output.

## Not verified / known limits
* Real sketches, real VLM output, real detectors: untested. Build the 30-50 case evaluation set (evaluate.py) first.
* Renderer is orthographic. Perspective sketches are approximated.
* Silhouette fit cannot see cuts, hidden depth, or global scale. Cuts and feature depths are always `assumed`.
* Extrusion profiles are taken from the VLM as-is (not refined against the image). Next step: optimise profile vertices.
* Features are attached to a part's AABB face then ray-cast to the true surface; features on strongly curved or
  sloped surfaces are approximate.
* `verified` on a feature = visible, non-grazing, confident detector. Not a measurement of size or depth.
* Mass uses a SOLID Al6061 body and only appears when an anchor dimension sets the scale.
* Screws/dial flutes are cosmetic library shapes; text, seams and shading are not modelled.
