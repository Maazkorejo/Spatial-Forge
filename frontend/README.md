# Spatial Forge

A single-page 2D-to-3D CAD workspace frontend built with React, TypeScript, Vite, and Three.js. The browser-local workflow includes drawing intake, sample engineering parts, a 3D viewport, dimension controls, cut-feature history, and export downloads.

## Run locally

Requirements: Node.js and the pinned pnpm version in `package.json`.

```sh
pnpm install
pnpm dev:static
```

Run `pnpm check` for TypeScript diagnostics and `pnpm build:static` for the production frontend in `dist/public`.

## Workspace

The default route opens the Edit workspace with an F-35 example. Use the Upload, Edit, and Export stage controls in the header or left rail. The uploader accepts `.dxf`, `.png`, `.jpg`, and `.webp`. Sample cases are provided for an F-35, wing profile, turned shaft, bracket, butterfly plate, wheel, and engine housing.

Three.js builds local procedural preview meshes. Dimensions, displayed volume, feature history, shading, orbit/pan/zoom, and browser-side exports update in the current workspace. The standalone HTML export includes a small self-contained canvas viewer. `ASSETS.md` lists the sources of user-provided and sample-library artwork.

## Frontend scope

This repository is a browser-local frontend prototype, not an automated CAD reconstruction backend. Uploaded 2D files start a local preview flow; their outlines are not solved by a production geometry kernel. Model samples are procedural visual references. Cut operations are represented in preview state and feature history rather than applied as industrial-grade Boolean CAD features. STEP and other deliverables are generated from preview meshes and are not certified manufacturing outputs. Verify important geometry with a dedicated CAD application before production.

## Branding and source structure

The user-provided Spatial Forge logo is stored at `client/public/spatialforge-logo.webp`. The sample F-35 image is stored at `client/public/f35-at-edwards.jpg`. The single route manifest is `client/public/manus-routes.json`. Main UI state is in `client/src/pages/Home.tsx`; the CAD viewport, mesh geometry, and exporters are separated under `client/src/components/cad/`; shared workspace styles are in `client/src/index.css`.
