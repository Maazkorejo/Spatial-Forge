============================================================
SpatialForge — 2D CAD (DXF) Test Drawings
============================================================

Drag and drop any of these .dxf files into the SpatialForge web UI (http://127.0.0.1:8085/ui) 
or click 'Choose DXF' / 'Change DXF' to test deterministic 3D B-Rep reconstruction:

• 01_stepped_prismatic_block_3rd_angle.dxf
  - Description: Prismatic Stepped Block with Through-Hole & Pocket (Third Angle Projection, Scale 1:1, mm units). Expected: Edge F1 100%, Watertight B-Rep Solid.

• 02_stepped_turned_shaft_3rd_angle.dxf
  - Description: Precision Revolved Turned Shaft with Concentric Shoulders (Third Angle Projection, Scale 1:1, mm units). Expected: Edge F1 100%, Revolved Watertight B-Rep Solid.

• 03_multifeature_baseplate_1st_angle.dxf
  - Description: Complex Multi-Feature Machine Baseplate with Counterbores & Chamfers (First Angle Projection, European Standard, mm units). Expected: Edge F1 98.6%, Watertight Solid.

• 04_stepped_pocket_housing_1st_angle.dxf
  - Description: Stepped Pocket Enclosure Housing with Cavity (First Angle Projection, mm units). Expected: Edge F1 93.8%+, Watertight Solid.
