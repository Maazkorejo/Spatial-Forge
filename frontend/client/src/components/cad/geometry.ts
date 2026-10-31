import * as THREE from "three";
import type { AppliedCut, BackendMeshData, Dimensions, DisplayMode, ModelKind, ModelState } from "./types";

const shellColor = new THREE.Color("#a9bdd8");
const edgeColor = new THREE.Color("#566984");

function geometryFromMeshData(meshData: BackendMeshData, targetDimensions: Dimensions, baseDimensions?: Dimensions): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const verts = meshData.vertices;
  const faces = meshData.faces;

  const positions = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    positions[i * 3] = verts[i][0];
    positions[i * 3 + 1] = verts[i][1];
    positions[i * 3 + 2] = verts[i][2];
  }

  const indices = new Uint32Array(faces.length * 3);
  for (let i = 0; i < faces.length; i++) {
    indices[i * 3] = faces[i][0];
    indices[i * 3 + 1] = faces[i][1];
    indices[i * 3 + 2] = faces[i][2];
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();

  const bb = geometry.boundingBox || new THREE.Box3();
  const centerX = (bb.min.x + bb.max.x) / 2;
  const centerY = bb.min.y;
  const centerZ = (bb.min.z + bb.max.z) / 2;

  geometry.translate(-centerX, -centerY, -centerZ);

  const baseL = baseDimensions?.length || targetDimensions.length;
  const baseW = baseDimensions?.width || targetDimensions.width;
  const baseT = baseDimensions?.thickness || targetDimensions.thickness;

  const scaleX = baseW > 0 ? targetDimensions.width / baseW : 1;
  const scaleY = baseT > 0 ? targetDimensions.thickness / baseT : 1;
  const scaleZ = baseL > 0 ? targetDimensions.length / baseL : 1;

  if (Math.abs(scaleX - 1) > 0.001 || Math.abs(scaleY - 1) > 0.001 || Math.abs(scaleZ - 1) > 0.001) {
    geometry.scale(scaleX, scaleY, scaleZ);
    geometry.computeVertexNormals();
  }

  return geometry;
}

function sheetGeometry(outline: Array<[number, number]>, height: number, holes: Array<[number, number, number]> = []) {
  const shape = new THREE.Shape();
  shape.moveTo(outline[0][0], outline[0][1]);
  for (const [x, y] of outline.slice(1)) shape.lineTo(x, y);
  shape.closePath();
  for (const [x, y, radius] of holes) {
    const hole = new THREE.Path();
    hole.absellipse(x, y, radius, radius, 0, Math.PI * 2, true, 0);
    shape.holes.push(hole);
  }
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: height,
    bevelEnabled: height > 5,
    bevelSegments: 2,
    bevelSize: Math.min(0.8, height * 0.08),
    bevelThickness: Math.min(0.8, height * 0.08),
    curveSegments: 8,
  });
  geometry.rotateX(-Math.PI / 2);
  geometry.computeVertexNormals();
  return geometry;
}

function planform(length: number, width: number, kind: ModelKind, seedText: string) {
  const l = length / 2;
  const w = width / 2;
  if (kind === "jet") {
    return [[l, 0], [l * 0.52, w * 0.14], [l * 0.23, w * 0.48], [-l * 0.04, w * 0.5], [-l * 0.13, w * 0.18], [-l * 0.45, w * 0.28], [-l * 0.5, w * 0.23], [-l * 0.44, 0], [-l * 0.5, -w * 0.23], [-l * 0.45, -w * 0.28], [-l * 0.13, -w * 0.18], [-l * 0.04, -w * 0.5], [l * 0.23, -w * 0.48], [l * 0.52, -w * 0.14]] as Array<[number, number]>;
  }
  if (kind === "wing") {
    return [[l, 0], [-l * 0.58, w], [-l, w], [-l * 0.62, 0], [-l, -w], [-l * 0.58, -w]] as Array<[number, number]>;
  }
  if (kind === "imported") {
    const seed = Array.from(seedText).reduce((value, character) => (value * 31 + character.charCodeAt(0)) >>> 0, 7);
    const count = 9;
    return Array.from({ length: count }, (_, index) => {
      const angle = (index / count) * Math.PI * 2;
      const step = ((seed >>> (index * 3 % 24)) % 19) / 100;
      const radius = index % 3 === 0 ? 0.44 + step : 0.22 + step;
      return [Math.cos(angle) * l * radius * 2.1, Math.sin(angle) * w * radius * 2.1] as [number, number];
    });
  }
  return [];
}

function makeSheet(model: ModelState) {
  const { length, width, thickness } = model.dimensions;
  const holes: Array<[number, number, number]> = [];
  const outline = planform(length, width, model.kind, model.title);
  if (model.kind === "bracket") {
    outline.splice(0, outline.length, [-length * 0.42, -width * 0.5], [length * 0.34, -width * 0.5], [length * 0.5, -width * 0.33], [length * 0.5, width * 0.32], [length * 0.32, width * 0.5], [-length * 0.42, width * 0.5]);
    holes.push([-length * 0.22, 0, width * 0.13], [length * 0.22, 0, width * 0.13]);
  } else if (model.kind === "butterfly") {
    outline.splice(0, outline.length, [0, 0], [-length * 0.17, width * 0.17], [-length * 0.42, width * 0.45], [-length * 0.48, width * 0.26], [-length * 0.38, width * 0.02], [-length * 0.49, -width * 0.26], [-length * 0.39, -width * 0.46], [-length * 0.14, -width * 0.18], [0, 0], [length * 0.14, -width * 0.18], [length * 0.39, -width * 0.46], [length * 0.49, -width * 0.26], [length * 0.38, width * 0.02], [length * 0.48, width * 0.26], [length * 0.42, width * 0.45], [length * 0.17, width * 0.17]);
  }
  if (outline.length) return sheetGeometry(outline, thickness, holes);
  return null;
}

function material(mode: DisplayMode, color = shellColor) {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: mode === "solid" ? 0.42 : 0.22,
    roughness: mode === "solid" ? 0.34 : 0.48,
    wireframe: mode === "wireframe",
    side: THREE.DoubleSide,
  });
}

function addMesh(group: THREE.Group, geometry: THREE.BufferGeometry, mode: DisplayMode, color?: THREE.ColorRepresentation, withEdges = true) {
  const mesh = new THREE.Mesh(geometry, material(mode, color ? new THREE.Color(color) : shellColor));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
  if (mode === "shaded" && withEdges) {
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 34), new THREE.LineBasicMaterial({ color: edgeColor, transparent: true, opacity: 0.76 }));
    edges.position.copy(mesh.position);
    edges.rotation.copy(mesh.rotation);
    edges.scale.copy(mesh.scale);
    edges.userData.edgeOverlay = true;
    group.add(edges);
  }
  return mesh;
}

function addFins(group: THREE.Group, model: ModelState, mode: DisplayMode) {
  const { length, width, thickness } = model.dimensions;
  for (const sign of [-1, 1]) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(length * 0.18, thickness * 1.5, width * 0.08), material(mode, new THREE.Color("#829bbd")));
    fin.position.set(-length * 0.35, thickness * 1.1, sign * width * 0.12);
    fin.rotation.z = -sign * 0.18;
    fin.castShadow = true;
    group.add(fin);
  }
}

function addShaft(group: THREE.Group, model: ModelState, mode: DisplayMode) {
  const { length, width } = model.dimensions;
  const r = Math.min(width, model.dimensions.thickness) * 0.5;
  const points = [new THREE.Vector2(0, 0), new THREE.Vector2(r * 0.66, 0), new THREE.Vector2(r * 0.66, length * 0.12), new THREE.Vector2(r, length * 0.12), new THREE.Vector2(r, length * 0.72), new THREE.Vector2(r * 0.73, length * 0.72), new THREE.Vector2(r * 0.73, length), new THREE.Vector2(0, length)];
  const geometry = new THREE.LatheGeometry(points, 36);
  geometry.rotateZ(-Math.PI / 2);
  geometry.translate(-length / 2, r, 0);
  addMesh(group, geometry, mode);
}

function addWheel(group: THREE.Group, model: ModelState, mode: DisplayMode) {
  const diameter = Math.min(model.dimensions.width, model.dimensions.length);
  const tire = new THREE.Mesh(new THREE.TorusGeometry(diameter * 0.39, diameter * 0.12, 18, 48), material(mode, new THREE.Color("#53687e")));
  tire.rotation.y = Math.PI / 2;
  tire.position.y = diameter * 0.51;
  tire.castShadow = true;
  group.add(tire);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(diameter * 0.18, diameter * 0.18, model.dimensions.thickness * 0.82, 32), material(mode, new THREE.Color("#9bb0cb")));
  hub.rotation.z = Math.PI / 2;
  hub.position.y = diameter * 0.51;
  hub.castShadow = true;
  group.add(hub);
  for (let index = 0; index < 8; index += 1) {
    const spoke = new THREE.Mesh(new THREE.BoxGeometry(diameter * 0.08, diameter * 0.55, model.dimensions.thickness * 0.46), material(mode, new THREE.Color("#758ba8")));
    spoke.position.set(0, diameter * 0.51, 0);
    spoke.rotation.x = index * Math.PI / 4;
    spoke.castShadow = true;
    group.add(spoke);
  }
}

function addHousing(group: THREE.Group, model: ModelState, mode: DisplayMode) {
  const { length, width, thickness } = model.dimensions;
  const radius = Math.min(width, length) * 0.37;
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.9, thickness, 40, 1, false), material(mode, new THREE.Color("#8299b5")));
  barrel.position.y = thickness / 2 + 2;
  barrel.castShadow = true;
  group.add(barrel);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(radius, Math.max(1, thickness * 0.035), 8, 40), material(mode, new THREE.Color("#677d99")));
  rim.rotation.x = Math.PI / 2;
  rim.position.y = thickness + 2;
  group.add(rim);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(radius * 1.12, radius * 1.12, 4, 40), material(mode, new THREE.Color("#94a8c1")));
  base.position.y = 2;
  base.castShadow = true;
  group.add(base);
}

function addButterflyBody(group: THREE.Group, model: ModelState, mode: DisplayMode) {
  const body = new THREE.Mesh(new THREE.CylinderGeometry(model.dimensions.thickness * 0.2, model.dimensions.thickness * 0.28, model.dimensions.width * 0.62, 12), material(mode, new THREE.Color("#607994")));
  body.position.y = model.dimensions.thickness * 0.8;
  body.castShadow = true;
  group.add(body);
}

function addCutMarkers(group: THREE.Group, model: ModelState, mode: DisplayMode) {
  model.cuts.forEach((cut, index) => {
    const radius = Math.max(1.5, Math.min(model.dimensions.length, model.dimensions.width) * (cut.glyph === "hole" ? 0.045 : 0.027));
    const marker = new THREE.Mesh(new THREE.TorusGeometry(radius, Math.max(0.3, model.dimensions.thickness * 0.012), 6, 28), new THREE.MeshStandardMaterial({ color: "#506a92", roughness: 0.7, metalness: 0.1 }));
    marker.rotation.x = Math.PI / 2;
    marker.position.set(model.dimensions.length * (index % 2 === 0 ? 0.15 : -0.2), model.dimensions.thickness + 0.12, model.dimensions.width * (index % 2 === 0 ? 0.16 : -0.17));
    marker.userData.cutMarker = true;
    group.add(marker);
  });
}

export function createPartGroup(model: ModelState, mode: DisplayMode) {
  const group = new THREE.Group();
  group.name = model.title;
  const { length, width, thickness } = model.dimensions;

  if (model.meshData && model.meshData.vertices?.length && model.meshData.faces?.length) {
    const geom = geometryFromMeshData(model.meshData, model.dimensions, model.baseDimensions);
    addMesh(group, geom, mode);
    addCutMarkers(group, model, mode);
    group.userData.dimensions = { length, width, thickness };
    return group;
  }

  const sheet = makeSheet(model);
  if (sheet) addMesh(group, sheet, mode);
  if (model.kind === "jet") addFins(group, model, mode);
  if (model.kind === "shaft") addShaft(group, model, mode);
  if (model.kind === "wheel") addWheel(group, model, mode);
  if (model.kind === "housing") addHousing(group, model, mode);
  if (model.kind === "butterfly") addButterflyBody(group, model, mode);
  if (model.kind === "bracket") {
    const foot = new THREE.Mesh(new THREE.BoxGeometry(length * 0.72, thickness * 0.6, width * 0.12), material(mode, new THREE.Color("#8299b5")));
    foot.position.set(-length * 0.05, thickness * 0.55, -width * 0.15);
    foot.rotation.z = -0.12;
    foot.castShadow = true;
    group.add(foot);
  }
  if (model.kind === "wing") {
    const spar = new THREE.Mesh(new THREE.BoxGeometry(length * 0.8, thickness * 0.55, Math.max(2, width * 0.08)), material(mode, new THREE.Color("#819ab8")));
    spar.position.y = thickness * 0.55;
    spar.castShadow = true;
    group.add(spar);
  }
  addCutMarkers(group, model, mode);
  group.userData.dimensions = { length, width, thickness };
  return group;
}

export function disposePartGroup(group: THREE.Group) {
  group.traverse((object) => {
    if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
      object.geometry.dispose();
      const mats = Array.isArray(object.material) ? object.material : [object.material];
      mats.forEach((item) => item.dispose());
    }
  });
}

export function cutLabel(cut: AppliedCut) {
  return cut.label;
}
