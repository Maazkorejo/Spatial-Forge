# drawing_gen — synthetic 2D-drawing benchmark for SpatialForge (step 1 of the 2D-CAD pivot)

Generates random mechanical parts in CadQuery, projects them to three orthographic views with hidden-line
removal (OpenCascade HLR), lays them out as a real drawing sheet, writes DXF, and injects realistic dirty-file
defects. Ground truth (solid, true dimensions, clean entities with roles, layout) is saved next to every file.

    pip install -r requirements.txt
    python test_drawings.py                              # 38 checks, ~1 min
    python generate.py out 20 --levels clean,light,heavy # 60 DXF + STEP + gt.json
    python generate.py out 20 --hull                     # also store visual-hull IoU per part (slower)

| file | role |
|---|---|
| parts.py | random prismatic 2.5D parts (holes, slot, counterbore, pockets incl. bottom/hidden-only, notch, step, chamfer) and turned parts (steps, chamfers, cross hole, blind bore) |
| project.py | HLR three-view projection; visible vs hidden; suppresses hidden lines coincident with visible ones |
| draw.py | layout (first/third angle), scale 1:1 / 1:2 / 2:1, centre lines, DIMENSIONS with REAL values, title block |
| noise.py | gaps, fragmentation, overlaps, duplicates, exploded dashes, exploded dimensions, layer-0 collapse, missing units, missing scale text |
| dxf_io.py | DXF writer/reader (ezdxf) |
| metrics.py | edge_prf (point-sampled, fragmentation-proof), volume_iou (exact B-rep), hull_iou (visual hull), flag_stats |
| preview.py | DXF -> PNG |
| previews/, sample_cases/ | examples: 4 seeds x 3 noise levels |

## Measured findings (test_drawings.py, 40 random parts)
* Only **5 / 40** parts are fully determined by their three silhouettes (visual-hull IoU 0.997).
* Prismatic parts with blind features: hull IoU 0.982 mean (min 0.964): small in volume, but these are the features that matter.
* Turned parts: hull IoU **0.909** mean (min 0.862). A body of revolution never comes out of Front ∩ Top ∩ Side
  (rounded-square section). A dedicated revolve path is required, not optional.
* Dimension ground truth: drawn length / dimension value = scale. At 1:2 the DXF stores measurement 47.5 with text "95".

## What is NOT covered (be careful with conclusions)
* Noise is my MODEL of dirty files, not sampled from real ones. A pipeline that passes here can still fail on real DXFs.
  Add 10-20 real files as a held-out set.
* No fillets / tangent edges, splines, ellipses, section views, detail views, auxiliary views, assemblies, GD&T, hatch.
* Arcs come only from the obround slot. Axis-aligned views only. Standard sheet layouts only.
* DXF text codes (%%c = diameter, %%d, %%p) are written but not decoded by preview.py; the ingestion stage must decode them.
* Hull IoU is voxel-based (36^3): good to ~1%.
* Not built yet: ingestion/normalisation, line classification, view registration, reconstruction, reprojection check.
  This package is the yardstick for those.
