# Spatial Forge Frontend Outcomes

## 1. Persistent shell and stage navigation
- Single-page shell has a left navigation rail and top header.
- Header shows the active source filename, current stage, Upload/Edit/Export stage buttons, concise status, Open File, and Reconstruct actions.
- Active model state remains available while switching between stages.
- Visual system uses a grey background, light surfaces, modern fonts, icons, buttons, and compact widgets; no orange or green colors are used.

## 2. Upload stage
- Full-width drag-and-drop zone accepts `.dxf`, `.png`, `.jpg`, and `.webp` files.
- File selection and drop provide concise validation and a processing-loader state.
- Sample engineering case library provides compact one-click cards that load a reconstruction into the workspace.

## 3. Edit workspace
- Three-column layout includes feature tree/measured specs, dominant 3D-style viewport, and compact edit controls.
- Viewport includes engineering ground grid, model-focused view, shading controls, and concise factual statuses such as Converged, Needs review, and Watertight.
- Floating 2D blueprint PiP supports minimize, restore, and close interactions.
- Precision controls support make bigger/smaller, Length/Width/Thickness steppers, and live volume updates.
- Cut workflow provides selectable cut actions, applied-cut list, individual removal actions, and immediate visual/model updates.
- Undo, redo, and reset preserve consistent dimensions, cuts, volume, and viewport state.

## 4. Export stage
- Deliverable grid provides working download actions for STEP, STL, HTML, GLB, and DXF.
- Export UI stays clean, concise, and focused on file type, size/status, and action.
