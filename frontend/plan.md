# Spatial Forge — implementation plan

## Product and implementation

Build a single-page React/TypeScript CAD workspace in the existing Vite starter. The public interface has a compact left navigation rail, file breadcrumb/header, and Upload, Edit, and Export stages. Keep CAD work local to the browser: sample selection, drawing upload, reconstruction progress, editing, undo/redo, and export actions all operate on a preserved active model. Use a procedural Three.js preview model, seeded with the F-35 example, rather than replacing a selected model with a generic block. This is a self-contained frontend prototype; actual automated 2D interpretation and analytic B-Rep reconstruction require a geometry service/kernel not included in this frontend.

The Edit stage is the main workspace: measured feature tree at left, centered 3D viewport with floor grid/orbit/pan/zoom and shading modes, optional source drawing PiP, and compact model dimensions/edit controls at right. Model controls update the currently selected procedural preview and metadata immediately; history snapshots power undo, redo, and reset. Upload accepts DXF and supported raster drawing formats. Export choices generate browser-side download artifacts based on the active preview; avoid claiming external engineering validation that the frontend cannot perform. Use existing Lucide icons and UI utilities; add Three.js plus its official STL/GLTF exporters for the viewport and mesh export. No authentication, managed server, or database is needed.

### Project structure

- `client/src/pages/Home.tsx` — application state, stage switching, sample selection, upload, edits, history, and export coordination.
- `client/src/components/cad/` — viewport, procedural geometry, and local export helpers; keep each concern in its own module.
- `client/src/index.css` — global near-black charcoal visual system and responsive workspace styling.
- `client/public/manus-routes.json` — one-page route declaration for `/`.
- `plan.md` and `TODO.md` — approved design/architecture and traceable product outcomes.

## Design

- **Design movement:** precision industrial software with restrained neo-brutalist panel geometry and a calm, tactile CAD-station finish.
- **Core principles:** keep the model dominant; group controls by engineering intent; show dimensions and state with direct visual affordances; keep copy short and status factual.
- **Color philosophy:** a near-black charcoal app shell, layered graphite panels, and a dark viewport create a focused CAD-station feel without any white surfaces. Reserve the logo's cobalt/indigo for active actions, selection, and small status accents; use cool light-grey text and steel outlines for clarity. Exclude orange and green throughout.
- **Layout paradigm:** persistent narrow icon rail, one-line workflow header, and an asymmetrical three-column edit bench with the canvas given the largest uninterrupted area. Upload and export remain focused, task-specific surfaces rather than generic marketing pages.
- **Signature elements:** a cobalt hairline/selection treatment around active tools; a low-contrast coordinate grid under each model; compact monospaced dimension readouts.
- **Interaction philosophy:** direct manipulation, instant feedback, clear selected/disabled states, keyboard-accessible controls, and concise tooltips rather than instructional paragraphs.
- **Animation:** short 140–220 ms transitions for stage changes, menus, and control feedback; subtle procedural loading-wheel motion during reconstruction; no continuous decoration that competes with viewport rotation.
- **Typography system:** Manrope for interface labels and headings with system-sans fallbacks; IBM Plex Mono for measurements, axes, and file metadata. Use a compact, legible hierarchy and avoid oversized marketing copy.
- **Brand essence:** a precise drawing-to-model workstation for engineers and makers; **focused, capable, measured**.
- **Brand voice:** terse, factual, tool-oriented. Examples: “Drawing ready to reconstruct.” “Solid volume · 34.8 cm³”.
- **Wordmark & logo:** use the user-provided Spatial Forge cube-and-wordmark artwork in the rail/header, rendered in a high-contrast light treatment on the dark surface; do not redraw or replace it. Keep a compact monogram fallback only for collapsed/mobile navigation.
- **Signature brand color:** restrained cobalt blue sampled conceptually from the supplied logo, used as the key active-action color; no orange or green.

## Delivery boundaries

Keep this frontend focused on an interactive client-side demonstration. Do not imply that a browser-only preview computes true watertight B-Rep geometry or that generated artifacts have passed manufacturing validation. Create the requested ZIP from the completed project source, excluding dependency/build caches and including project instructions and lockfile.
